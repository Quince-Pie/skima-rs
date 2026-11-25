//! Cache-line aligned, expandable per-frame bump allocator implementing Rust 1.100.0's `std::alloc::Allocator`.

use std::alloc::{AllocError, Allocator, Layout};
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

const MIN_CHUNK_ALIGN: usize = 64;
const MIN_CHUNK_CAP: usize = 4096;

struct Chunk {
    ptr: NonNull<u8>,
    cap: usize,
    align: usize,
}

impl Chunk {
    fn new_aligned(cap: usize, align: usize) -> Result<Self, AllocError> {
        let align = align.max(MIN_CHUNK_ALIGN);
        if !align.is_power_of_two() {
            return Err(AllocError);
        }
        let cap = cap
            .max(MIN_CHUNK_CAP)
            .max(align)
            .checked_next_power_of_two()
            .ok_or(AllocError)?;
        let layout = Layout::from_size_align(cap, align).map_err(|_| AllocError)?;
        // SAFETY: `cap >= 4096` (> 0) and `align` is a valid power-of-two layout.
        let raw = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(raw).ok_or(AllocError)?;
        Ok(Self { ptr, cap, align })
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.cap, self.align).expect("valid chunk layout");
        // SAFETY: Allocated with the identical `(self.cap, self.align)` layout in `Chunk::new_aligned`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// Fast multi-chunk bump arena implementing `std::alloc::Allocator` for `&FrameArena`.
///
/// - Fast path allocates from the active `>= 64`-byte-aligned chunk in ~3 instructions (two `Cell` ops).
/// - Supports arbitrary power-of-two alignments (including `> 64` bytes such as AVX-512 / page-aligned types)
///   and zero-sized types (`ZST`s) with strict pointer provenance preservation.
/// - In-place `grow`, `grow_zeroed`, `shrink`, and LIFO `deallocate` when modifying the most recently
///   allocated buffer (`Vec<T, &FrameArena>`).
/// - `reset()` consolidates capacity (and max alignment) if overflow chunks were allocated, dropping
///   steady-state heap allocations to **zero** after warmup without imposing any fixed capacity limit.
pub struct FrameArena {
    active_ptr: Cell<NonNull<u8>>,
    active_cap: Cell<usize>,
    active_align: Cell<usize>,
    offset: Cell<usize>,
    chunks: RefCell<Vec<Chunk>>,
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::with_capacity(256 * 1024)
    }
}

impl FrameArena {
    pub fn with_capacity(initial_cap: usize) -> Self {
        let chunk = Chunk::new_aligned(initial_cap, MIN_CHUNK_ALIGN).unwrap_or_else(|_| {
            std::alloc::handle_alloc_error(
                Layout::from_size_align(MIN_CHUNK_CAP, MIN_CHUNK_ALIGN).unwrap(),
            )
        });
        let ptr = chunk.ptr;
        let cap = chunk.cap;
        let align = chunk.align;
        Self {
            active_ptr: Cell::new(ptr),
            active_cap: Cell::new(cap),
            active_align: Cell::new(align),
            offset: Cell::new(0),
            chunks: RefCell::new(vec![chunk]),
        }
    }

    /// Reset the arena for the next frame in O(1).
    /// If the previous frame spilled into multiple chunks, consolidates into a single right-sized chunk
    /// preserving both total capacity and maximum alignment so subsequent frames never spill.
    #[inline]
    pub fn reset(&self) {
        let mut chunks = self.chunks.borrow_mut();
        if chunks.len() > 1 {
            let total_cap = chunks
                .iter()
                .fold(0usize, |acc, c| acc.saturating_add(c.cap));
            let max_align = chunks
                .iter()
                .map(|c| c.align)
                .max()
                .unwrap_or(MIN_CHUNK_ALIGN);
            if let Ok(consolidated) = Chunk::new_aligned(total_cap, max_align) {
                let ptr = consolidated.ptr;
                let cap = consolidated.cap;
                let align = consolidated.align;
                chunks.clear();
                chunks.push(consolidated);
                self.active_ptr.set(ptr);
                self.active_cap.set(cap);
                self.active_align.set(align);
            } else {
                // Retain the largest already-allocated chunk on OOM fallback rather than the smallest.
                let mut best_idx = 0usize;
                for i in 1..chunks.len() {
                    if (chunks[i].cap, chunks[i].align) > (chunks[best_idx].cap, chunks[best_idx].align) {
                        best_idx = i;
                    }
                }
                if best_idx != 0 {
                    chunks.swap(0, best_idx);
                }
                chunks.truncate(1);
                self.active_ptr.set(chunks[0].ptr);
                self.active_cap.set(chunks[0].cap);
                self.active_align.set(chunks[0].align);
            }
        }
        self.offset.set(0);
    }

    #[inline]
    pub fn used_bytes_in_active_chunk(&self) -> usize {
        self.offset.get()
    }

    #[inline]
    pub fn total_capacity(&self) -> usize {
        self.chunks
            .borrow()
            .iter()
            .fold(0usize, |acc, c| acc.saturating_add(c.cap))
    }

    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunks.borrow().len()
    }

    /// Allocate and copy a slice directly into the frame arena, returning a frame-lived slice `&[T]`.
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> &[T] {
        if src.is_empty() || core::mem::size_of::<T>() == 0 {
            // SAFETY: `NonNull::dangling()` is properly aligned and valid for zero-byte slices
            // (both `src.is_empty()` and arbitrary-length ZST slices).
            return unsafe { core::slice::from_raw_parts(NonNull::<T>::dangling().as_ptr(), src.len()) };
        }
        let layout = Layout::for_value(src);
        let raw = self
            .allocate(layout)
            .unwrap_or_else(|_| std::alloc::handle_alloc_error(layout));
        let dst = raw.cast::<T>();
        // SAFETY: `raw` was freshly allocated from `self`'s active chunk with `layout` matching `src`,
        // carries direct chunk provenance, and does not overlap `src`.
        unsafe {
            dst.as_ptr().copy_from_nonoverlapping(src.as_ptr(), src.len());
            core::slice::from_raw_parts(dst.as_ptr(), src.len())
        }
    }

    #[cold]
    fn alloc_slow(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let req = layout
            .size()
            .max(layout.align())
            .checked_next_power_of_two()
            .ok_or(AllocError)?;
        let next_cap = self.active_cap.get().saturating_mul(2).max(req);
        let next_align = self.active_align.get().max(layout.align());
        let new_chunk = Chunk::new_aligned(next_cap, next_align)?;
        let ptr = new_chunk.ptr;
        let cap = new_chunk.cap;
        let align = new_chunk.align;
        self.chunks.borrow_mut().push(new_chunk);
        self.active_ptr.set(ptr);
        self.active_cap.set(cap);
        self.active_align.set(align);
        self.offset.set(layout.size());
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    /// Returns `Some(rel_offset)` iff `ptr` (for a non-zero-sized allocation) lies strictly inside
    /// `[active_ptr .. active_ptr + active_cap)`.
    ///
    /// Strict `< active_cap` is essential under Rust pointer provenance / Miri: if an older spilled
    /// chunk happens to sit immediately after the active chunk at `active_ptr + active_cap`, a non-ZST
    /// allocation at offset 0 of that older chunk would have `rel == active_cap` and MUST NOT be
    /// re-derived from `active_ptr`!
    #[inline]
    fn active_rel_offset(&self, ptr: NonNull<u8>) -> Option<usize> {
        let base = self.active_ptr.get().addr().get();
        let addr = ptr.addr().get();
        let rel = addr.checked_sub(base)?;
        (rel < self.active_cap.get()).then_some(rel)
    }
}

unsafe impl Allocator for &FrameArena {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }
        let align = layout.align();
        let mask = align - 1;
        let cur = self.offset.get();
        if align <= self.active_align.get() {
            // Fast path: chunk base is already a multiple of `align`, so aligning `offset` suffices.
            if let Some(padded) = cur.checked_add(mask) {
                let aligned = padded & !mask;
                if let Some(end) = aligned.checked_add(layout.size())
                    && end <= self.active_cap.get()
                {
                    self.offset.set(end);
                    // SAFETY: `aligned + layout.size() <= active_cap`, within the active `Chunk` allocation.
                    let ptr = unsafe { self.active_ptr.get().byte_add(aligned) };
                    return Ok(NonNull::slice_from_raw_parts(ptr, layout.size()));
                }
            }
        } else {
            // Over-aligned request (`align > active_align`): align the physical address within the chunk if room permits.
            let base = self.active_ptr.get().addr().get();
            if let Some(cur_addr) = base.checked_add(cur)
                && let Some(padded_addr) = cur_addr.checked_add(mask)
            {
                let aligned_addr = padded_addr & !mask;
                let aligned_rel = aligned_addr - base;
                if let Some(end) = aligned_rel.checked_add(layout.size())
                    && end <= self.active_cap.get()
                {
                    self.offset.set(end);
                    // SAFETY: `aligned_rel + layout.size() <= active_cap`, derived from `active_ptr` provenance.
                    let ptr = unsafe { self.active_ptr.get().byte_add(aligned_rel) };
                    return Ok(NonNull::slice_from_raw_parts(ptr, layout.size()));
                }
            }
        }
        core::hint::cold_path();
        self.alloc_slow(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // Reclaim LIFO top-of-bump immediately if `ptr` was the most recent allocation.
        if let Some(rel) = self.active_rel_offset(ptr)
            && rel <= self.offset.get()
            && self.offset.get() - rel == layout.size()
        {
            self.offset.set(rel);
        }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        if old_layout.size() == 0 {
            return self.allocate(new_layout);
        }
        if let Some(rel) = self.active_rel_offset(ptr)
            && rel <= self.offset.get()
            && self.offset.get() - rel == old_layout.size()
            && (ptr.addr().get() & (new_layout.align() - 1)) == 0
            && let Some(new_end) = rel.checked_add(new_layout.size())
            && new_end <= self.active_cap.get()
        {
            self.offset.set(new_end);
            // SAFETY: Re-derive from `active_ptr` so the returned slice carries full chunk provenance.
            let chunk_ptr = unsafe { self.active_ptr.get().byte_add(rel) };
            return Ok(NonNull::slice_from_raw_parts(chunk_ptr, new_layout.size()));
        }
        let new_mem = self.allocate(new_layout)?;
        // SAFETY: Caller guarantees `ptr` is valid for `old_layout.size() > 0` bytes and non-overlapping with `new_mem`.
        unsafe {
            new_mem
                .cast::<u8>()
                .copy_from_nonoverlapping(ptr, old_layout.size());
        }
        Ok(new_mem)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() >= old_layout.size());
        if old_layout.size() == 0 {
            return self.allocate_zeroed(new_layout);
        }
        if let Some(rel) = self.active_rel_offset(ptr)
            && rel <= self.offset.get()
            && self.offset.get() - rel == old_layout.size()
            && (ptr.addr().get() & (new_layout.align() - 1)) == 0
            && let Some(new_end) = rel.checked_add(new_layout.size())
            && new_end <= self.active_cap.get()
        {
            let delta = new_layout.size() - old_layout.size();
            // SAFETY: `rel + new_layout.size() <= active_cap`, so tail `[rel + old..rel + new)` is valid writable memory.
            let chunk_ptr = unsafe { self.active_ptr.get().byte_add(rel) };
            unsafe {
                chunk_ptr.byte_add(old_layout.size()).write_bytes(0, delta);
            }
            self.offset.set(new_end);
            return Ok(NonNull::slice_from_raw_parts(chunk_ptr, new_layout.size()));
        }
        let new_mem = self.allocate_zeroed(new_layout)?;
        // SAFETY: Caller guarantees `ptr` is valid for `old_layout.size() > 0` bytes and non-overlapping with `new_mem`.
        unsafe {
            new_mem
                .cast::<u8>()
                .copy_from_nonoverlapping(ptr, old_layout.size());
        }
        Ok(new_mem)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(new_layout.size() <= old_layout.size());
        if new_layout.size() == 0 {
            // SAFETY: Caller upholds `deallocate` preconditions for `(ptr, old_layout)`.
            unsafe { self.deallocate(ptr, old_layout) };
            return Ok(NonNull::slice_from_raw_parts(new_layout.dangling_ptr(), 0));
        }
        if (ptr.addr().get() & (new_layout.align() - 1)) == 0 {
            if let Some(rel) = self.active_rel_offset(ptr) {
                if rel <= self.offset.get() && self.offset.get() - rel == old_layout.size() {
                    self.offset.set(rel + new_layout.size());
                }
                // SAFETY: `rel < active_cap` and `rel + new_layout.size() <= rel + old_layout.size() <= active_cap`.
                let chunk_ptr = unsafe { self.active_ptr.get().byte_add(rel) };
                return Ok(NonNull::slice_from_raw_parts(chunk_ptr, new_layout.size()));
            }
            // `ptr` belongs to an older spilled chunk: retain `ptr`'s own provenance over `new_layout.size() <= old_layout.size()`.
            return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
        }
        let new_mem = self.allocate(new_layout)?;
        // SAFETY: `new_layout.size() <= old_layout.size()` and `new_mem` is freshly allocated.
        unsafe {
            new_mem
                .cast::<u8>()
                .copy_from_nonoverlapping(ptr, new_layout.size());
        }
        Ok(new_mem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(256))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Aligned256([u64; 32]);

    #[repr(C, align(4096))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct PageAligned([u8; 4096]);

    #[repr(align(8192))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct HugeAlignZst;

    #[test]
    fn zst_and_empty_vec_grow_preserve_provenance_and_zero_usage() {
        let arena = FrameArena::with_capacity(4096);
        let a = &arena;
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);

        let mut zvec: Vec<HugeAlignZst, &FrameArena> = Vec::new_in(a);
        for _ in 0..1000 {
            zvec.push(HugeAlignZst);
        }
        assert_eq!(zvec.len(), 1000);
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);

        let zslice = arena.alloc_slice_copy(&[HugeAlignZst; 32]);
        assert_eq!(zslice.len(), 32);
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);

        // Fill active chunk to 4096 bytes, then grow a fresh empty Vec<PageAligned, &FrameArena>
        // whose dangling pointer (4096) might numerically coincide with `offset == 4096`.
        let _pad = a.allocate(Layout::from_size_align(4096, 64).unwrap()).unwrap();
        assert_eq!(arena.used_bytes_in_active_chunk(), 4096);

        let mut pvec: Vec<PageAligned, &FrameArena> = Vec::new_in(a);
        pvec.push(PageAligned([0xAB; 4096]));
        assert_eq!(pvec[0].0[0], 0xAB);
        assert_eq!(pvec[0].0[4095], 0xAB);
        assert_eq!(pvec.as_ptr().addr() % 4096, 0);
    }

    #[test]
    fn over_aligned_allocations_and_spillover_consolidation() {
        let arena = FrameArena::with_capacity(4096);
        let a = &arena;

        let b1 = a.allocate(Layout::from_size_align(7, 1).unwrap()).unwrap();
        assert_eq!(b1.len(), 7);

        let a256 = a
            .allocate(Layout::from_size_align(256, 256).unwrap())
            .unwrap();
        assert_eq!(a256.cast::<u8>().as_ptr().addr() % 256, 0);

        let a4096 = a
            .allocate(Layout::from_size_align(8192, 4096).unwrap())
            .unwrap();
        assert_eq!(a4096.cast::<u8>().as_ptr().addr() % 4096, 0);

        // Spill into multiple chunks, then consolidate on `reset()`.
        let mut v = Vec::with_capacity_in(64, a);
        for i in 0..64u64 {
            v.push(Aligned256([i; 32]));
        }
        assert_eq!(v.as_ptr().addr() % 256, 0);
        drop(v);

        arena.reset();
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);
        let cap_after_reset = arena.total_capacity();
        assert!(cap_after_reset >= 64 * 256 + 8192);

        // Subsequent frame with identical workload must never spill or grow capacity.
        let p2 = a
            .allocate(Layout::from_size_align(4096, 4096).unwrap())
            .unwrap();
        assert_eq!(p2.cast::<u8>().as_ptr().addr() % 4096, 0);
        let mut v2 = Vec::with_capacity_in(64, a);
        for i in 0..64u64 {
            v2.push(Aligned256([i; 32]));
        }
        assert_eq!(arena.total_capacity(), cap_after_reset);
    }

    #[test]
    fn in_place_lifo_grow_zeroed_shrink_and_deallocate() {
        let arena = FrameArena::with_capacity(16384);
        let a = &arena;

        let mut v: Vec<u32, &FrameArena> = Vec::with_capacity_in(16, a);
        let first_ptr = v.as_ptr();
        assert_eq!(arena.used_bytes_in_active_chunk(), 64);

        // In-place `grow`: pointer stays identical!
        v.reserve_exact(64); // 64 u32s = 256B
        assert_eq!(v.as_ptr(), first_ptr);
        let grown_used = arena.used_bytes_in_active_chunk();
        assert_eq!(grown_used, 256);

        // In-place `shrink`: reclaims top-of-bump bytes without moving pointer!
        v.extend(0..8);
        v.shrink_to_fit();
        assert_eq!(v.as_ptr(), first_ptr);
        assert_eq!(arena.used_bytes_in_active_chunk(), 32);

        // LIFO `deallocate`: dropping top-of-bump Vec resets `used` to 0!
        drop(v);
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);

        // Verify `grow_zeroed` zeroes newly exposed tail bytes in-place (even over dirty arena memory):
        {
            let dirty = a.allocate(Layout::from_size_align(512, 8).unwrap()).unwrap();
            unsafe { dirty.cast::<u8>().write_bytes(0xFF, 512) };
            unsafe { a.deallocate(dirty.cast::<u8>(), Layout::from_size_align(512, 8).unwrap()) };
        }
        assert_eq!(arena.used_bytes_in_active_chunk(), 0);

        let l_small = Layout::from_size_align(64, 8).unwrap();
        let l_large = Layout::from_size_align(256, 8).unwrap();
        let mem0 = a.allocate_zeroed(l_small).unwrap();
        let mem1 = unsafe { a.grow_zeroed(mem0.cast::<u8>(), l_small, l_large).unwrap() };
        assert_eq!(mem0.cast::<u8>(), mem1.cast::<u8>());
        let bytes = unsafe { core::slice::from_raw_parts(mem1.cast::<u8>().as_ptr(), 256) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn spilled_chunk_shrink_grow_and_adjacent_boundary_provenance_isolation() {
        let arena = FrameArena::with_capacity(4096);
        let a = &arena;
        assert_eq!(arena.chunk_count(), 1);

        // Allocate inside Chunk 0:
        let mut old_vec: Vec<u64, &FrameArena> = Vec::with_capacity_in(32, a);
        old_vec.extend(0..16);

        // Force spill into Chunk 1 (`8192` bytes):
        let big = a.allocate(Layout::from_size_align(6000, 64).unwrap()).unwrap();
        assert_eq!(arena.chunk_count(), 2);
        assert_eq!(big.len(), 6000);

        // `shrink_to_fit` on `old_vec` (residing in older spilled Chunk 0!) must preserve
        // Chunk 0 provenance and data without touching Chunk 1's active offset!
        let active_used_before = arena.used_bytes_in_active_chunk();
        let old_ptr = old_vec.as_ptr();
        old_vec.shrink_to_fit();
        assert_eq!(old_vec.as_ptr(), old_ptr);
        assert_eq!(arena.used_bytes_in_active_chunk(), active_used_before);
        for (i, &val) in old_vec.iter().enumerate() {
            assert_eq!(val, i as u64);
        }

        // Growing `old_vec` from spilled Chunk 0 relocates cleanly into active Chunk 1:
        old_vec.reserve_exact(64);
        assert_ne!(old_vec.as_ptr(), old_ptr);
        for (i, &val) in old_vec.iter().enumerate() {
            assert_eq!(val, i as u64);
        }

        // Verify strict `< active_cap` boundary guard: a pointer sitting right at
        // `active_ptr + active_cap` (simulating an adjacent older chunk immediately above
        // the active chunk in virtual memory) MUST NOT be classified as inside the active chunk!
        let end_of_active = unsafe { arena.active_ptr.get().byte_add(arena.active_cap.get()) };
        assert_eq!(arena.active_rel_offset(end_of_active), None);

        // Verify direct `alloc_slice_copy` on empty, ZST, and aligned slices:
        let empty: &[u64] = arena.alloc_slice_copy(&[]);
        assert!(empty.is_empty());
        let zsts = arena.alloc_slice_copy(&[HugeAlignZst; 128]);
        assert_eq!(zsts.len(), 128);
        let copied = arena.alloc_slice_copy(&[Aligned256([7; 32]), Aligned256([9; 32])]);
        assert_eq!(copied.len(), 2);
        assert_eq!(copied.as_ptr().addr() % 256, 0);
        assert_eq!(copied[0].0[0], 7);
        assert_eq!(copied[1].0[31], 9);
    }
}
