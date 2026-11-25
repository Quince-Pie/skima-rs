//! Typed generational pools (`SlotPool<T>`), multi-type `Realm`, smart projections,
//! and closed-form analytical spring physics (`Spring`).

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::DropGuard;
use std::slice::GetDisjointMutError;

/// Compact 8-byte `Copy` generational handle into `SlotPool<T>`.
/// Unlike GPUI's `Entity<T>` (backed by heap `Box<dyn Any>` and runtime lease removal),
/// `Key<T>` indexes a dense contiguous `Vec<SlotEntry<T>>` with zero `TypeId` downcasts.
///
/// Live keys always carry `generation` in `1..=u32::MAX` (never `0`).
pub struct Key<T> {
    pub(crate) index: u32,
    pub(crate) generation: u32,
    _marker: PhantomData<fn() -> T>,
}

const _: () = assert!(core::mem::size_of::<Key<u64>>() == 8);

impl<T> Key<T> {
    /// Sentinel invalid key (`generation == 0`) that is guaranteed never to match any live slot.
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
        _marker: PhantomData,
    };

    #[inline]
    pub const fn is_null(&self) -> bool {
        self.generation == 0
    }
}

impl<T> Clone for Key<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Key<T> {}

impl<T> PartialEq for Key<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Key<T> {}

impl<T> std::hash::Hash for Key<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> std::fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Key({}v{})", self.index, self.generation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

struct SlotEntry<T> {
    value: Option<T>,
    generation: u32,
    epoch: u64,
    subscribers: Vec<ScopeId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolAccessError {
    StaleOrInvalidKey,
    OverlappingKeys(GetDisjointMutError),
}

#[inline]
fn push_unique_scope(dirty_scopes: &mut Vec<ScopeId>, scope: ScopeId) {
    if !dirty_scopes.contains(&scope) {
        dirty_scopes.push(scope);
    }
}

/// Dense typed generational pool for shared/graph domain state (`Buffer`, `Document`, `AudioGraph`).
///
/// Generation safety invariant:
/// - Live slots strictly use `generation` in `1..=u32::MAX` (`generation == 0` is reserved for
///   permanently retired / invalid keys).
/// - If a single slot reaches `generation == u32::MAX` and is removed, it is permanently retired
///   at `generation = 0` rather than recycled onto `free_head`, mathematically eliminating both
///   `generation == 0` live keys and any wraparound ABA collision.
pub struct SlotPool<T> {
    slots: Vec<SlotEntry<T>>,
    free_head: Vec<u32>,
    dirty_scopes: Vec<ScopeId>,
    len: usize,
}

impl<T> Default for SlotPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SlotPool<T> {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: Vec::new(),
            dirty_scopes: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let capped = capacity.min(u32::MAX as usize);
        Self {
            slots: Vec::with_capacity(capped),
            free_head: Vec::new(),
            dirty_scopes: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total slot capacity currently allocated without reallocating the backing `Vec`
    /// (capped at `u32::MAX as usize`).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity().min(u32::MAX as usize)
    }

    /// Reserve capacity for at least `additional` more live items to be inserted without reallocating.
    pub fn reserve(&mut self, additional: usize) {
        let reusable = self.free_head.len();
        if additional > reusable {
            let need = additional - reusable;
            let max_extra = (u32::MAX as usize).saturating_sub(self.slots.len());
            self.slots.reserve(need.min(max_extra));
        }
    }

    /// Remove all live items while strictly advancing (or retiring at `u32::MAX -> 0`) every
    /// cleared slot's generation so previously issued `Key<T>` handles can NEVER suffer ABA collision.
    ///
    /// Also notifies any subscribed `ScopeId`s and preserves exact pool invariants if a `T::drop` panics.
    pub fn clear(&mut self) {
        for idx in 0..self.slots.len() {
            let slot = &mut self.slots[idx];
            if slot.value.is_some() {
                for &sub in &slot.subscribers {
                    push_unique_scope(&mut self.dirty_scopes, sub);
                }
                slot.subscribers.clear();
                self.len -= 1;
                if slot.generation < u32::MAX {
                    slot.generation += 1;
                    self.free_head.push(idx as u32);
                } else {
                    slot.generation = 0;
                }
                let removed = slot.value.take();
                drop(removed);
            }
        }
    }

    /// Retain only the live entries satisfying `f(key, &mut value)`, removing all others while
    /// preserving generation wraparound retirement, reactive `dirty_scopes`, and unwind safety.
    pub fn retain(&mut self, mut f: impl FnMut(Key<T>, &mut T) -> bool) {
        for idx in 0..self.slots.len() {
            let slot = &mut self.slots[idx];
            if slot.generation == 0 || slot.value.is_none() {
                continue;
            }
            let key = Key {
                index: idx as u32,
                generation: slot.generation,
                _marker: PhantomData,
            };
            slot.epoch = slot.epoch.wrapping_add(1).max(1);
            for &sub in &slot.subscribers {
                push_unique_scope(&mut self.dirty_scopes, sub);
            }
            let keep = f(key, slot.value.as_mut().unwrap());
            if !keep {
                slot.subscribers.clear();
                self.len -= 1;
                if slot.generation < u32::MAX {
                    slot.generation += 1;
                    self.free_head.push(idx as u32);
                } else {
                    slot.generation = 0;
                }
                let removed = slot.value.take();
                drop(removed);
            }
        }
    }

    /// Iterate over all live `(Key<T>, &T)` pairs in dense slot order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.slots.iter().enumerate(),
            remaining: self.len,
        }
    }

    /// Mutably iterate over all live `(Key<T>, &mut T)` pairs in dense slot order,
    /// advancing `epoch` and recording `dirty_scopes` for each yielded slot.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            iter: self.slots.iter_mut().enumerate(),
            dirty_scopes: &mut self.dirty_scopes,
            remaining: self.len,
        }
    }

    /// Iterate over all live `Key<T>` handles in dense slot order.
    #[inline]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = Key<T>> + DoubleEndedIterator + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over all live `&T` values in dense slot order.
    #[inline]
    pub fn values(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator + '_ {
        self.iter().map(|(_, v)| v)
    }

    /// Mutably iterate over all live `&mut T` values in dense slot order.
    #[inline]
    pub fn values_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut T> + DoubleEndedIterator + '_ {
        self.iter_mut().map(|(_, v)| v)
    }

    pub fn insert(&mut self, value: T) -> Key<T> {
        while let Some(idx) = self.free_head.pop() {
            let slot = &mut self.slots[idx as usize];
            if slot.generation == 0 {
                // Retired slot guard: never issue generation 0.
                continue;
            }
            slot.value = Some(value);
            slot.epoch = slot.epoch.wrapping_add(1).max(1);
            slot.subscribers.clear();
            self.len += 1;
            return Key {
                index: idx,
                generation: slot.generation,
                _marker: PhantomData,
            };
        }

        assert!(
            self.slots.len() < (u32::MAX as usize),
            "SlotPool capacity exhausted"
        );
        let idx = self.slots.len() as u32;
        self.slots.push(SlotEntry {
            value: Some(value),
            generation: 1,
            epoch: 1,
            subscribers: Vec::new(),
        });
        self.len += 1;
        Key {
            index: idx,
            generation: 1,
            _marker: PhantomData,
        }
    }

    pub fn remove(&mut self, key: Key<T>) -> Option<T> {
        if key.generation == 0 {
            return None;
        }
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation || slot.value.is_none() {
            return None;
        }
        for &sub in &slot.subscribers {
            push_unique_scope(&mut self.dirty_scopes, sub);
        }
        slot.subscribers.clear();
        self.len -= 1;

        if slot.generation < u32::MAX {
            slot.generation += 1;
            self.free_head.push(key.index);
        } else {
            // Exhausted all `2^32 - 1` non-zero generations on this slot: retire permanently at 0
            // so `generation == 0` is never issued and no historical `Key` can ever suffer ABA collision.
            slot.generation = 0;
        }
        slot.value.take()
    }

    #[inline]
    pub fn contains(&self, key: Key<T>) -> bool {
        key.generation != 0
            && self
                .slots
                .get(key.index as usize)
                .is_some_and(|s| s.generation == key.generation && s.value.is_some())
    }

    #[inline]
    pub fn get(&self, key: Key<T>) -> Option<&T> {
        if key.generation == 0 {
            return None;
        }
        let slot = self.slots.get(key.index as usize)?;
        (slot.generation == key.generation)
            .then_some(slot.value.as_ref())
            .flatten()
    }

    #[inline]
    pub fn read_tracked(&mut self, key: Key<T>, scope: ScopeId) -> Option<(&T, u64)> {
        if key.generation == 0 {
            return None;
        }
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        let val = slot.value.as_ref()?;
        if !slot.subscribers.contains(&scope) {
            slot.subscribers.push(scope);
        }
        Some((val, slot.epoch))
    }

    #[inline]
    pub fn unsubscribe(&mut self, slot_index: u32, scope: ScopeId) {
        if let Some(slot) = self.slots.get_mut(slot_index as usize)
            && let Some(pos) = slot.subscribers.iter().position(|&s| s == scope)
        {
            slot.subscribers.swap_remove(pos);
        }
    }

    #[inline]
    pub fn subscriber_count(&self, key: Key<T>) -> usize {
        if key.generation == 0 {
            return 0;
        }
        self.slots
            .get(key.index as usize)
            .filter(|s| s.generation == key.generation && s.value.is_some())
            .map_or(0, |s| s.subscribers.len())
    }

    #[inline]
    pub fn epoch(&self, key: Key<T>) -> Option<u64> {
        if key.generation == 0 {
            return None;
        }
        let slot = self.slots.get(key.index as usize)?;
        (slot.generation == key.generation && slot.value.is_some()).then_some(slot.epoch)
    }

    #[inline]
    pub fn get_mut(&mut self, key: Key<T>) -> Option<&mut T> {
        if key.generation == 0 {
            return None;
        }
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation || slot.value.is_none() {
            return None;
        }
        slot.epoch = slot.epoch.wrapping_add(1).max(1);
        for &sub in &slot.subscribers {
            push_unique_scope(&mut self.dirty_scopes, sub);
        }
        slot.value.as_mut()
    }

    /// Simultaneous disjoint mutable borrows of `N` entities of the SAME type (`Rust 1.86+`).
    ///
    /// Atomic validation invariant: if ANY key is overlapping, out-of-bounds, stale, or invalid,
    /// returns `Err` without mutating any slot's `epoch` or emitting any `dirty_scopes`.
    pub fn get_disjoint_mut<const N: usize>(
        &mut self,
        keys: [Key<T>; N],
    ) -> Result<[&mut T; N], PoolAccessError> {
        let indices: [usize; N] = keys.map(|k| k.index as usize);
        let mut entries = match self.slots.get_disjoint_mut(indices) {
            Ok(entries) => entries,
            Err(GetDisjointMutError::OverlappingIndices) => {
                return Err(PoolAccessError::OverlappingKeys(
                    GetDisjointMutError::OverlappingIndices,
                ));
            }
            Err(GetDisjointMutError::IndexOutOfBounds) => {
                for i in 0..N {
                    for j in (i + 1)..N {
                        if indices[i] == indices[j] {
                            return Err(PoolAccessError::OverlappingKeys(
                                GetDisjointMutError::OverlappingIndices,
                            ));
                        }
                    }
                }
                return Err(PoolAccessError::StaleOrInvalidKey);
            }
        };

        for (entry, key) in entries.iter().zip(keys.iter()) {
            if key.generation == 0 || entry.generation != key.generation || entry.value.is_none() {
                return Err(PoolAccessError::StaleOrInvalidKey);
            }
        }

        for entry in &mut entries {
            entry.epoch = entry.epoch.wrapping_add(1).max(1);
            for &sub in &entry.subscribers {
                push_unique_scope(&mut self.dirty_scopes, sub);
            }
        }

        Ok(entries.map(|e| e.value.as_mut().unwrap()))
    }

    /// Transactional mutation via Rust 1.100.0 `core::mem::DropGuard`.
    /// Rolls back `T` on `Err` or panic without incrementing `epoch` or emitting dirty notifications.
    pub fn try_update_transactional<E>(
        &mut self,
        key: Key<T>,
        f: impl FnOnce(&mut T) -> Result<(), E>,
    ) -> Result<bool, E>
    where
        T: Clone,
    {
        if key.generation == 0 {
            return Ok(false);
        }
        let Some(slot) = self.slots.get_mut(key.index as usize) else {
            return Ok(false);
        };
        if slot.generation != key.generation || slot.value.is_none() {
            return Ok(false);
        }
        let backup = slot.value.as_ref().unwrap().clone();
        let val_ref = slot.value.as_mut().unwrap();
        let mut guard = DropGuard::new((val_ref, backup), |(target, old)| {
            *target = old;
        });

        f(guard.0)?;

        let _ = DropGuard::dismiss(guard);
        slot.epoch = slot.epoch.wrapping_add(1).max(1);
        for &sub in &slot.subscribers {
            push_unique_scope(&mut self.dirty_scopes, sub);
        }
        Ok(true)
    }

    #[inline]
    pub fn drain_dirty_scopes(&mut self, out: &mut Vec<ScopeId>) {
        out.append(&mut self.dirty_scopes);
    }
}

pub struct Iter<'a, T> {
    iter: core::iter::Enumerate<core::slice::Iter<'a, SlotEntry<T>>>,
    remaining: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Key<T>, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        for (idx, slot) in self.iter.by_ref() {
            if slot.generation != 0
                && let Some(val) = slot.value.as_ref()
            {
                self.remaining -= 1;
                return Some((
                    Key {
                        index: idx as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    val,
                ));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while let Some((idx, slot)) = self.iter.next_back() {
            if slot.generation != 0
                && let Some(val) = slot.value.as_ref()
            {
                self.remaining -= 1;
                return Some((
                    Key {
                        index: idx as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    val,
                ));
            }
        }
        None
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}
impl<T> core::iter::FusedIterator for Iter<'_, T> {}

pub struct IterMut<'a, T> {
    iter: core::iter::Enumerate<core::slice::IterMut<'a, SlotEntry<T>>>,
    dirty_scopes: &'a mut Vec<ScopeId>,
    remaining: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (Key<T>, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        for (idx, slot) in self.iter.by_ref() {
            if slot.generation != 0
                && let Some(val) = slot.value.as_mut()
            {
                self.remaining -= 1;
                slot.epoch = slot.epoch.wrapping_add(1).max(1);
                for &sub in &slot.subscribers {
                    push_unique_scope(self.dirty_scopes, sub);
                }
                return Some((
                    Key {
                        index: idx as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    val,
                ));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while let Some((idx, slot)) = self.iter.next_back() {
            if slot.generation != 0
                && let Some(val) = slot.value.as_mut()
            {
                self.remaining -= 1;
                slot.epoch = slot.epoch.wrapping_add(1).max(1);
                for &sub in &slot.subscribers {
                    push_unique_scope(self.dirty_scopes, sub);
                }
                return Some((
                    Key {
                        index: idx as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    val,
                ));
            }
        }
        None
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}
impl<T> core::iter::FusedIterator for IterMut<'_, T> {}

impl<'a, T> IntoIterator for &'a SlotPool<T> {
    type Item = (Key<T>, &'a T);
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut SlotPool<T> {
    type Item = (Key<T>, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

trait ErasedPool: Any {
    fn drain_dirty(&mut self, out: &mut Vec<ScopeId>);
    fn unsubscribe_erased(&mut self, slot_index: u32, scope: ScopeId);
}

impl<T: 'static> ErasedPool for SlotPool<T> {
    fn drain_dirty(&mut self, out: &mut Vec<ScopeId>) {
        self.drain_dirty_scopes(out);
    }
    fn unsubscribe_erased(&mut self, slot_index: u32, scope: ScopeId) {
        self.unsubscribe(slot_index, scope);
    }
}

/// Type-indexed multi-pool `Realm` supporting simultaneous disjoint `&mut SlotPool<A>` and `&mut SlotPool<B>`
/// via `HashMap::get_disjoint_mut` (stabilized in Rust 1.86!).
#[derive(Default)]
pub struct Realm {
    pools: HashMap<TypeId, Box<dyn ErasedPool>>,
}

impl Realm {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    pub fn pool<T: 'static>(&self) -> Option<&SlotPool<T>> {
        let erased = self.pools.get(&TypeId::of::<T>())?;
        // Rust 1.86 trait upcasting: `&dyn ErasedPool` -> `&dyn Any` directly!
        (&**erased as &dyn Any).downcast_ref::<SlotPool<T>>()
    }

    pub fn pool_mut<T: 'static>(&mut self) -> &mut SlotPool<T> {
        let entry = self
            .pools
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(SlotPool::<T>::new()));
        (&mut **entry as &mut dyn Any)
            .downcast_mut::<SlotPool<T>>()
            .expect("TypeId matches SlotPool<T>")
    }

    /// Borrow TWO distinct typed pools (`&mut SlotPool<A>` and `&mut SlotPool<B>`) simultaneously!
    /// Uses Rust 1.86's `HashMap::get_disjoint_mut` + Trait Upcasting (`&mut dyn ErasedPool as &mut dyn Any`).
    pub fn pools_disjoint_mut<A: 'static, B: 'static>(
        &mut self,
    ) -> (&mut SlotPool<A>, &mut SlotPool<B>) {
        let tid_a = TypeId::of::<A>();
        let tid_b = TypeId::of::<B>();
        assert_ne!(
            tid_a, tid_b,
            "Use SlotPool::get_disjoint_mut for same-type entities"
        );
        self.pools
            .entry(tid_a)
            .or_insert_with(|| Box::new(SlotPool::<A>::new()));
        self.pools
            .entry(tid_b)
            .or_insert_with(|| Box::new(SlotPool::<B>::new()));

        let [Some(pa), Some(pb)] = self.pools.get_disjoint_mut([&tid_a, &tid_b]) else {
            unreachable!("distinct TypeIds were just inserted");
        };
        let a = (&mut **pa as &mut dyn Any)
            .downcast_mut::<SlotPool<A>>()
            .unwrap();
        let b = (&mut **pb as &mut dyn Any)
            .downcast_mut::<SlotPool<B>>()
            .unwrap();
        (a, b)
    }

    pub fn unsubscribe_by_type(&mut self, tid: TypeId, slot_index: u32, scope: ScopeId) {
        if let Some(p) = self.pools.get_mut(&tid) {
            p.unsubscribe_erased(slot_index, scope);
        }
    }

    pub fn drain_all_dirty_scopes(&mut self, out: &mut Vec<ScopeId>) {
        for p in self.pools.values_mut() {
            p.drain_dirty(out);
        }
    }
}

/// Closed-form analytical damped harmonic spring (`x(t)` solved in `f64` for arbitrary `dt`,
/// zero numerical Euler instability across 60Hz/120Hz/240Hz, extreme damping ratios, or frame drops).
#[derive(Clone, Copy, Debug)]
pub struct Spring {
    pub value: f32,
    pub velocity: f32,
    pub target: f32,
    pub angular_freq: f32,  // omega_0 (rad/s), e.g., 22.0 for snappy UI
    pub damping_ratio: f32, // zeta: 1.0 = critically damped, < 1.0 = bouncy, > 1.0 = overdamped
}

impl Spring {
    pub const fn new(initial: f32, response_hz: f32, damping_ratio: f32) -> Self {
        Self {
            value: initial,
            velocity: 0.0,
            target: initial,
            angular_freq: response_hz * core::f32::consts::TAU,
            damping_ratio,
        }
    }

    pub const fn critically_damped(initial: f32, response_hz: f32) -> Self {
        Self::new(initial, response_hz, 1.0)
    }

    pub const fn underdamped(initial: f32, response_hz: f32, damping_ratio: f32) -> Self {
        Self::new(initial, response_hz, damping_ratio)
    }

    pub const fn overdamped(initial: f32, response_hz: f32, damping_ratio: f32) -> Self {
        Self::new(initial, response_hz, damping_ratio)
    }

    #[inline]
    pub fn set_target(&mut self, target: f32) {
        if target.is_finite() {
            self.target = target;
        }
    }

    #[inline]
    pub fn is_settled(&self, eps: f32) -> bool {
        self.value.is_finite()
            && self.velocity.is_finite()
            && self.target.is_finite()
            && (self.value - self.target).abs() <= eps
            && self.velocity.abs() <= eps
    }

    /// Advance by exact closed-form differential equation solution over `dt` seconds.
    /// Evaluates internally in `f64` across all three damping regimes (`zeta < 1`, `zeta == 1`, `zeta > 1`),
    /// avoiding catastrophic root cancellation at large `zeta` and `Inf * 0` / `cos(Inf)` overflow at huge `dt`.
    /// Returns `true` while still moving, `false` once settled on `target`.
    pub fn step(&mut self, dt: f32) -> bool {
        if !self.target.is_finite() {
            self.target = if self.value.is_finite() {
                self.value
            } else {
                0.0
            };
        }
        if !self.value.is_finite() {
            self.value = self.target;
            self.velocity = 0.0;
            return false;
        }
        if !self.velocity.is_finite() {
            self.velocity = 0.0;
        }
        if dt.is_nan() || dt <= 0.0 {
            if self.is_settled(1e-4) {
                self.value = self.target;
                self.velocity = 0.0;
                return false;
            }
            return true;
        }
        let z = if self.damping_ratio.is_finite() {
            (self.damping_ratio as f64).max(0.0)
        } else {
            1.0
        };
        if dt.is_infinite() {
            if z > 0.0 {
                self.value = self.target;
                self.velocity = 0.0;
                return false;
            }
            return !self.is_settled(1e-4);
        }

        let x0 = (self.value as f64) - (self.target as f64);
        let v0 = self.velocity as f64;
        if x0.abs() <= 1e-4 && v0.abs() <= 1e-4 {
            self.value = self.target;
            self.velocity = 0.0;
            return false;
        }
        let w = if self.angular_freq.is_finite() {
            (self.angular_freq as f64).max(0.1)
        } else {
            core::f64::consts::TAU
        };
        let t = dt as f64;

        let (x1, v1) = if (z - 1.0).abs() < 1e-5 {
            // Critically damped exact solution: x(t) = (x0 + (v0 + w*x0)*t) * e^(-w*t)
            let wt = w * t;
            if wt > 700.0 {
                (0.0, 0.0)
            } else {
                let exp = (-wt).exp();
                let c2 = v0 + w * x0;
                let nx = (x0 + c2 * t) * exp;
                let nv = (v0 - c2 * wt) * exp;
                (nx, nv)
            }
        } else if z < 1.0 {
            // Underdamped (and undamped z == 0) exact solution:
            let zwt = z * w * t;
            if zwt > 700.0 {
                (0.0, 0.0)
            } else {
                let wd = w * (1.0 - z * z).sqrt();
                let exp = (-zwt).exp();
                let phase = (wd * t).rem_euclid(core::f64::consts::TAU);
                let (sin, cos) = phase.sin_cos();
                let c2 = (v0 + z * w * x0) / wd;
                let nx = exp * (x0 * cos + c2 * sin);
                let nv = exp * (v0 * cos - (x0 * wd + z * w * c2) * sin);
                (nx, nv)
            }
        } else {
            // Overdamped exact solution (`z > 1.0`):
            // Rationalize `r1 = -w / (z + sqrt(z^2 - 1))` to prevent catastrophic cancellation at large `z`!
            let s = (z * z - 1.0).sqrt();
            let zp = z + s;
            let r1 = -w / zp;
            let r2 = -w * zp;
            let neg_r1_t = -r1 * t;
            if neg_r1_t > 700.0 {
                (0.0, 0.0)
            } else {
                let c2 = (v0 - r1 * x0) / (-2.0 * w * s);
                let c1 = x0 - c2;
                let e1 = (-neg_r1_t).exp();
                let neg_r2_t = -r2 * t;
                let e2 = if neg_r2_t > 700.0 {
                    0.0
                } else {
                    (-neg_r2_t).exp()
                };
                (c1 * e1 + c2 * e2, c1 * r1 * e1 + c2 * r2 * e2)
            }
        };

        let next_val = (self.target as f64 + x1) as f32;
        let next_vel = v1 as f32;
        if !next_val.is_finite() || !next_vel.is_finite() {
            self.value = self.target;
            self.velocity = 0.0;
            return false;
        }
        self.value = next_val;
        self.velocity = next_vel;
        if self.is_settled(1e-4) {
            self.value = self.target;
            self.velocity = 0.0;
            false
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_pool_generation_wraparound_never_emits_zero_or_aba_collision() {
        let mut pool = SlotPool::<u32>::new();
        let k_initial = pool.insert(100);
        assert_eq!(k_initial.index, 0);
        assert_eq!(k_initial.generation, 1);
        assert!(!pool.contains(Key::INVALID));

        // Simulate a slot sitting at `u32::MAX - 1` about to hit `u32::MAX` and then retire.
        pool.slots[0].generation = u32::MAX - 1;
        let stale_v1 = k_initial;
        let cur_key = Key {
            index: 0,
            generation: u32::MAX - 1,
            _marker: PhantomData,
        };
        assert_eq!(pool.remove(cur_key), Some(100));
        assert_eq!(pool.len(), 0);

        // Next insert reuses slot 0 at `generation == u32::MAX`.
        let k_max = pool.insert(200);
        assert_eq!(k_max.index, 0);
        assert_eq!(k_max.generation, u32::MAX);
        assert_eq!(pool.get(k_max), Some(&200));

        // Removing `k_max` (`generation == u32::MAX`) MUST retire slot 0 at generation 0
        // rather than wrapping to 0 or 1!
        assert_eq!(pool.remove(k_max), Some(200));
        assert_eq!(pool.slots[0].generation, 0);
        assert!(!pool.contains(k_max));
        assert!(!pool.contains(stale_v1));
        assert!(
            !pool.contains(Key {
                index: 0,
                generation: 0,
                _marker: PhantomData,
            })
        );

        // Next insert allocates fresh slot 1 with generation 1: zero ABA collision with `stale_v1` or `k_max`!
        let k_fresh = pool.insert(300);
        assert_eq!(k_fresh.index, 1);
        assert_eq!(k_fresh.generation, 1);
        assert_eq!(pool.get(stale_v1), None);
        assert_eq!(pool.get(k_max), None);
        assert_eq!(pool.get(k_fresh), Some(&300));
    }

    #[test]
    fn get_disjoint_mut_enforces_all_or_nothing_and_distinguishes_errors() {
        let mut pool = SlotPool::<i32>::new();
        let k0 = pool.insert(10);
        let k1 = pool.insert(20);
        let k2 = pool.insert(30);

        let _ = pool.read_tracked(k0, ScopeId(7));
        let ep0_before = pool.epoch(k0).unwrap();

        // 1. Overlapping live keys -> OverlappingKeys
        assert!(matches!(
            pool.get_disjoint_mut([k0, k0]),
            Err(PoolAccessError::OverlappingKeys(_))
        ));

        // 2. Out-of-bounds foreign key -> StaleOrInvalidKey (NOT OverlappingKeys!)
        let oob_key = Key {
            index: 999,
            generation: 1,
            _marker: PhantomData,
        };
        assert_eq!(
            pool.get_disjoint_mut([k0, oob_key]),
            Err(PoolAccessError::StaleOrInvalidKey)
        );

        // 3. Stale removed key paired with valid key -> StaleOrInvalidKey, and `k0` epoch/dirty untouched!
        pool.remove(k2);
        let mut dirty = Vec::new();
        pool.drain_dirty_scopes(&mut dirty);
        dirty.clear();

        assert_eq!(
            pool.get_disjoint_mut([k0, k2]),
            Err(PoolAccessError::StaleOrInvalidKey)
        );
        assert_eq!(pool.epoch(k0), Some(ep0_before));
        pool.drain_dirty_scopes(&mut dirty);
        assert!(dirty.is_empty());

        // 4. Valid disjoint mutation updates values, epochs, and deduplicated dirty scopes:
        let [v0, v1] = pool.get_disjoint_mut([k0, k1]).unwrap();
        *v0 += 5;
        *v1 += 15;
        assert_eq!(pool.get(k0), Some(&15));
        assert_eq!(pool.get(k1), Some(&35));
        assert!(pool.epoch(k0).unwrap() > ep0_before);
        pool.drain_dirty_scopes(&mut dirty);
        assert_eq!(dirty, vec![ScopeId(7)]);
    }

    #[test]
    fn transactional_update_rolls_back_on_err_and_panic() {
        let mut pool = SlotPool::<Vec<i32>>::new();
        let k = pool.insert(vec![1, 2, 3]);
        let _ = pool.read_tracked(k, ScopeId(42));
        let ep0 = pool.epoch(k).unwrap();

        let res: Result<bool, &str> = pool.try_update_transactional(k, |v| {
            v.push(99);
            Err("abort")
        });
        assert_eq!(res, Err("abort"));
        assert_eq!(pool.get(k).unwrap(), &[1, 2, 3]);
        assert_eq!(pool.epoch(k), Some(ep0));

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<bool, ()> = pool.try_update_transactional(k, |v| {
                v.clear();
                panic!("boom");
            });
        }));
        assert!(unwind.is_err());
        assert_eq!(pool.get(k).unwrap(), &[1, 2, 3]);
        assert_eq!(pool.epoch(k), Some(ep0));
        let mut dirty = Vec::new();
        pool.drain_dirty_scopes(&mut dirty);
        assert!(dirty.is_empty());
    }

    #[test]
    fn spring_all_three_regimes_semigroup_invariance_and_extreme_stability() {
        for &zeta in &[0.35_f32, 1.0_f32, 2.5_f32, 50_000.0_f32] {
            let mut s_one = Spring::new(0.0, 3.0, zeta);
            s_one.set_target(100.0);
            s_one.step(0.1);

            let mut s_many = Spring::new(0.0, 3.0, zeta);
            s_many.set_target(100.0);
            for _ in 0..100 {
                s_many.step(0.001);
            }
            assert!(
                (s_one.value - s_many.value).abs() < 1e-3,
                "zeta={zeta}: 1x0.1s ({}) vs 100x0.001s ({})",
                s_one.value,
                s_many.value
            );
            assert!(
                s_one.value > 0.0,
                "zeta={zeta} must make forward progress (no overdamped stall)"
            );
        }

        // Huge `dt`, `Infinity`, `<= 0`, and `NaN` stability across all 3 regimes:
        for &zeta in &[0.2_f32, 1.0_f32, 8.0_f32] {
            let mut s = Spring::new(-50.0, 4.0, zeta);
            s.set_target(250.0);
            assert!(s.step(0.0));
            assert!(s.step(-1.0));
            assert!(s.step(f32::NAN));
            assert_eq!(s.value, -50.0);

            assert!(!s.step(1e38));
            assert_eq!(s.value, 250.0);
            assert_eq!(s.velocity, 0.0);

            s.value = 10.0;
            assert!(!s.step(f32::INFINITY));
            assert_eq!(s.value, 250.0);
            assert_eq!(s.velocity, 0.0);
        }
    }

    #[test]
    fn slot_pool_iter_retain_clear_capacity_and_generation_unwind_safety() {
        let mut pool = SlotPool::<i32>::with_capacity(16);
        assert!(pool.capacity() >= 16);
        assert!(pool.is_empty());

        let k0 = pool.insert(10);
        let k1 = pool.insert(21);
        let k2 = pool.insert(30);
        let k3 = pool.insert(45);
        assert_eq!(pool.len(), 4);

        let _ = pool.read_tracked(k0, ScopeId(1));
        let _ = pool.read_tracked(k1, ScopeId(2));
        let _ = pool.read_tracked(k2, ScopeId(3));

        // Remove middle slot `k1` so `slots` has an interior hole:
        assert_eq!(pool.remove(k1), Some(21));
        assert_eq!(pool.len(), 3);
        let mut dirty = Vec::new();
        pool.drain_dirty_scopes(&mut dirty);
        assert_eq!(dirty, vec![ScopeId(2)]);
        dirty.clear();

        // Forward & reverse ExactSizeIterator on `iter()` / `keys()` / `values()`:
        let fwd: Vec<(Key<i32>, i32)> = (&pool).into_iter().map(|(k, &v)| (k, v)).collect();
        assert_eq!(fwd, vec![(k0, 10), (k2, 30), (k3, 45)]);
        let rev: Vec<i32> = pool.values().rev().copied().collect();
        assert_eq!(rev, vec![45, 30, 10]);
        assert_eq!(pool.keys().len(), 3);

        // `iter_mut()` updates values, increments epochs, and emits dirty scopes only for live slots:
        let ep0 = pool.epoch(k0).unwrap();
        for (_k, v) in &mut pool {
            *v += 1;
        }
        assert_eq!(pool.get(k0), Some(&11));
        assert_eq!(pool.get(k2), Some(&31));
        assert_eq!(pool.get(k3), Some(&46));
        assert!(pool.epoch(k0).unwrap() > ep0);
        pool.drain_dirty_scopes(&mut dirty);
        assert!(dirty.contains(&ScopeId(1)) && dirty.contains(&ScopeId(3)));
        dirty.clear();

        // `retain`: keep only even values (`k3 = 46`), retiring `k0` and `k2` safely:
        let _ = pool.read_tracked(k0, ScopeId(9));
        pool.retain(|_k, v| (*v % 2) == 0);
        assert_eq!(pool.len(), 1);
        assert!(!pool.contains(k0));
        assert!(!pool.contains(k2));
        assert_eq!(pool.get(k3), Some(&46));
        pool.drain_dirty_scopes(&mut dirty);
        assert!(dirty.contains(&ScopeId(9)));
        dirty.clear();

        // Force `k3`'s slot to `u32::MAX` before `clear()` to verify `clear()` retires `u32::MAX -> 0`
        // AND increments normal slots so NO historical `Key<T>` ever collides after `clear()`!
        let k_reused = pool.insert(99);
        assert_ne!(k_reused, k0);
        assert_ne!(k_reused, k1);
        assert_ne!(k_reused, k2);
        let _ = pool.read_tracked(k_reused, ScopeId(77));
        pool.slots[k3.index as usize].generation = u32::MAX;
        let k3_max = Key {
            index: k3.index,
            generation: u32::MAX,
            _marker: PhantomData,
        };

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.iter().count(), 0);
        pool.drain_dirty_scopes(&mut dirty);
        assert_eq!(dirty, vec![ScopeId(77)]);

        // Re-insert 8 fresh items after `clear()` and verify ZERO collisions with any pre-clear key:
        pool.reserve(8);
        let post_keys: Vec<_> = (0..8).map(|i| pool.insert(i * 100)).collect();
        for &old_k in &[k0, k1, k2, k3, k3_max, k_reused, Key::INVALID] {
            assert!(!pool.contains(old_k), "stale key {old_k:?} collided after clear!");
            assert_eq!(pool.get(old_k), None);
        }
        assert_eq!(post_keys.len(), 8);
        assert_eq!(pool.len(), 8);

        // Unwind safety of `clear()` when `T::drop` panics mid-pool:
        struct Bomb(bool);
        impl Drop for Bomb {
            fn drop(&mut self) {
                if self.0 {
                    panic!("drop panic");
                }
            }
        }
        let mut bpool = SlotPool::<Bomb>::new();
        let b0 = bpool.insert(Bomb(false));
        let b1 = bpool.insert(Bomb(true));
        let b2 = bpool.insert(Bomb(false));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bpool.clear();
        }));
        assert!(res.is_err());
        assert!(!bpool.contains(b0));
        assert!(!bpool.contains(b1));
        assert!(bpool.contains(b2));
        assert_eq!(bpool.len(), 1);
        bpool.clear();
        assert!(bpool.is_empty());
    }
}
