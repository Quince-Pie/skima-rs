//! Incremental `accesskit 0.24` Integration.
//! Emits minimal `O(dirty_a11y_nodes)` `accesskit::TreeUpdate`s when assistive technology is active
//! and performs **zero** allocations when screen readers are inactive.

use crate::layout::ComputedBox;
pub use accesskit::{
    Action, ActionData, ActionRequest, Affine as A11yAffine, Node as A11yNode,
    NodeId as A11yNodeId, Rect as A11yRect, Role as A11yRole, Toggled, Tree as A11yTree, TreeId,
    TreeUpdate,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticsMeta {
    pub role: A11yRole,
    pub label: Option<String>,
    pub value: Option<String>,
    pub numeric_value: Option<f64>,
    pub min_numeric_value: Option<f64>,
    pub max_numeric_value: Option<f64>,
    pub numeric_step: Option<f64>,
    pub toggled: Option<Toggled>,
    pub clickable: bool,
    pub focusable: bool,
    pub incrementable: bool,
}

impl Default for SemanticsMeta {
    fn default() -> Self {
        Self {
            role: A11yRole::GenericContainer,
            label: None,
            value: None,
            numeric_value: None,
            min_numeric_value: None,
            max_numeric_value: None,
            numeric_step: None,
            toggled: None,
            clickable: false,
            focusable: false,
            incrementable: false,
        }
    }
}

impl SemanticsMeta {
    pub fn button(label: impl Into<String>) -> Self {
        Self {
            role: A11yRole::Button,
            label: Some(label.into()),
            clickable: true,
            focusable: true,
            ..Self::default()
        }
    }

    pub fn checkbox(label: impl Into<String>, checked: bool) -> Self {
        Self {
            role: A11yRole::CheckBox,
            label: Some(label.into()),
            toggled: Some(if checked { Toggled::True } else { Toggled::False }),
            clickable: true,
            focusable: true,
            ..Self::default()
        }
    }

    pub fn switch(label: impl Into<String>, enabled: bool) -> Self {
        Self {
            role: A11yRole::Switch,
            label: Some(label.into()),
            toggled: Some(if enabled { Toggled::True } else { Toggled::False }),
            clickable: true,
            focusable: true,
            ..Self::default()
        }
    }

    pub fn slider(label: impl Into<String>, val: f64) -> Self {
        Self {
            role: A11yRole::Slider,
            label: Some(label.into()),
            numeric_value: Some(val),
            focusable: true,
            incrementable: true,
            ..Self::default()
        }
    }

    pub fn slider_ranged(label: impl Into<String>, val: f64, min: f64, max: f64, step: f64) -> Self {
        Self {
            role: A11yRole::Slider,
            label: Some(label.into()),
            numeric_value: Some(val),
            min_numeric_value: Some(min),
            max_numeric_value: Some(max),
            numeric_step: Some(step),
            focusable: true,
            incrementable: true,
            ..Self::default()
        }
    }

    pub fn scrollbar(label: impl Into<String>, val: f64) -> Self {
        Self {
            role: A11yRole::ScrollBar,
            label: Some(label.into()),
            numeric_value: Some(val),
            focusable: true,
            clickable: true,
            incrementable: true,
            ..Self::default()
        }
    }

    pub fn scrollbar_ranged(label: impl Into<String>, val: f64, max: f64) -> Self {
        Self {
            role: A11yRole::ScrollBar,
            label: Some(label.into()),
            numeric_value: Some(val),
            min_numeric_value: Some(0.0),
            max_numeric_value: Some(max.max(0.0)),
            numeric_step: Some(38.0),
            focusable: true,
            clickable: true,
            incrementable: true,
            ..Self::default()
        }
    }

    pub fn scroll_view(label: impl Into<String>, scroll_y: f64, max_scroll: f64) -> Self {
        Self {
            role: A11yRole::ScrollView,
            label: Some(label.into()),
            numeric_value: Some(scroll_y),
            min_numeric_value: Some(0.0),
            max_numeric_value: Some(max_scroll.max(0.0)),
            focusable: true,
            incrementable: max_scroll > 0.0,
            ..Self::default()
        }
    }

    pub fn progress_indicator(fraction: f64) -> Self {
        Self {
            role: A11yRole::ProgressIndicator,
            numeric_value: Some(fraction.clamp(0.0, 1.0)),
            min_numeric_value: Some(0.0),
            max_numeric_value: Some(1.0),
            ..Self::default()
        }
    }

    pub fn text_input(label: impl Into<String>, current_value: impl Into<String>) -> Self {
        Self {
            role: A11yRole::TextInput,
            label: Some(label.into()),
            value: Some(current_value.into()),
            focusable: true,
            clickable: true,
            ..Self::default()
        }
    }

    pub fn tooltip(label: impl Into<String>) -> Self {
        Self {
            role: A11yRole::Tooltip,
            label: Some(label.into()),
            ..Self::default()
        }
    }

    pub fn build_accesskit_node(&self, bounds: ComputedBox, child_ids: &[A11yNodeId]) -> A11yNode {
        let mut node = A11yNode::new(self.role);
        node.set_bounds(A11yRect {
            x0: bounds.x as f64,
            y0: bounds.y as f64,
            x1: (bounds.x + bounds.w) as f64,
            y1: (bounds.y + bounds.h) as f64,
        });
        if let Some(lbl) = &self.label {
            node.set_label(lbl.as_str());
        }
        if let Some(val) = &self.value {
            node.set_value(val.as_str());
        }
        if let Some(num) = self.numeric_value {
            node.set_numeric_value(num);
        }
        if let Some(min_v) = self.min_numeric_value {
            node.set_min_numeric_value(min_v);
        }
        if let Some(max_v) = self.max_numeric_value {
            node.set_max_numeric_value(max_v);
        }
        if let Some(step_v) = self.numeric_step {
            node.set_numeric_value_step(step_v);
        }
        if let Some(tog) = self.toggled {
            node.set_toggled(tog);
        }
        if self.clickable {
            node.add_action(Action::Click);
        }
        if self.focusable {
            node.add_action(Action::Focus);
        }
        if self.incrementable || matches!(self.role, A11yRole::Slider | A11yRole::ScrollBar) {
            node.add_action(Action::Increment);
            node.add_action(Action::Decrement);
        }
        if self.role == A11yRole::TextInput {
            node.add_action(Action::SetValue);
        }
        if !child_ids.is_empty() {
            node.set_children(child_ids.to_vec());
        }
        node
    }
}

/// Apply the window's HiDPI logical-to-physical scale transform onto the root `Role::Window`
/// `AccessKit` node when `scale_factor != 1.0` (`0 B` allocation at `1.0x`).
#[inline]
pub fn apply_root_scale_transform(node: &mut A11yNode, scale_factor: f32) {
    if scale_factor.is_finite() && scale_factor > 0.0 && (scale_factor - 1.0).abs() > 1e-5 {
        node.set_transform(A11yAffine::scale(scale_factor as f64));
    } else {
        node.clear_transform();
    }
}

#[inline]
pub const fn slot_to_a11y_id(slot_idx: u32) -> A11yNodeId {
    // Non-zero stable ID (+1 so slot 0 maps to `#1`)
    A11yNodeId((slot_idx as u64) + 1)
}

#[inline]
pub const fn a11y_id_to_slot(id: A11yNodeId) -> Option<u32> {
    if id.0 == 0 {
        None
    } else {
        Some((id.0 - 1) as u32)
    }
}
