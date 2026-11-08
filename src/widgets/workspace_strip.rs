//! A row (or column) of full-size pages, each shifted along `axis` by a
//! fractional number of pages. Used to show the workspaces next to the
//! active one while a touchpad swipe moves between them.
//!
//! Children are laid out at the full size of the strip, then translated by
//! `offset * (extent + gap)`. Nothing is clipped: capture images are
//! subsurfaces, which can't be clipped by the renderer anyway, and the strip
//! moves perpendicular to the workspace bar, so a page that slides out only
//! goes off the edge of the output.
//!
//! Children are drawn with the parent's viewport unchanged: widening it to
//! cover the off-screen pages left the whole layer surface blank afterwards
//! on the GLES renderer.

use cosmic::iced::{
    Length, Point, Rectangle, Size,
    advanced::{
        Clipboard, Layout, Shell, Widget,
        layout::{self, flex::Axis},
        mouse, renderer,
        widget::{Operation, Tree},
    },
    event::Event,
};

const GAP: f32 = 24.0;

pub fn workspace_strip<'a, Msg>(
    pages: Vec<(f32, cosmic::Element<'a, Msg>)>,
    axis: Axis,
    width: Length,
    height: Length,
) -> WorkspaceStrip<'a, Msg> {
    let (offsets, children) = pages.into_iter().unzip();
    WorkspaceStrip {
        offsets,
        children,
        axis,
        width,
        height,
    }
}

pub struct WorkspaceStrip<'a, Msg> {
    offsets: Vec<f32>,
    children: Vec<cosmic::Element<'a, Msg>>,
    axis: Axis,
    width: Length,
    height: Length,
}

impl<Msg> Widget<Msg, cosmic::Theme, cosmic::Renderer> for WorkspaceStrip<'_, Msg> {
    fn size(&self) -> Size<Length> {
        Size::new(self.width, self.height)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &cosmic::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(self.width, self.height, Size::ZERO);
        let child_limits = layout::Limits::new(Size::ZERO, size);
        let axis = &self.axis;
        let children = self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(&self.offsets)
            .map(|((child, tree), offset)| {
                let node = child.as_widget_mut().layout(tree, renderer, &child_limits);
                let position = match axis {
                    Axis::Horizontal => Point::new(offset * (size.width + GAP), 0.0),
                    Axis::Vertical => Point::new(0.0, offset * (size.height + GAP)),
                };
                node.move_to(position)
            })
            .collect();
        layout::Node::with_children(size, children)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &cosmic::Renderer,
        operation: &mut dyn Operation<()>,
    ) {
        for ((child, tree), layout) in self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(layout.children())
        {
            child.as_widget_mut().operate(tree, layout, renderer, operation);
        }
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &cosmic::Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Msg>,
        viewport: &Rectangle,
    ) {
        for ((child, tree), layout) in self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(layout.children())
        {
            child.as_widget_mut().update(
                tree, event, layout, cursor, renderer, clipboard, shell, viewport,
            );
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &cosmic::Renderer,
    ) -> mouse::Interaction {
        self.children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
            .map(|((child, tree), layout)| {
                child
                    .as_widget()
                    .mouse_interaction(tree, layout, cursor, viewport, renderer)
            })
            .max()
            .unwrap_or_default()
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut cosmic::Renderer,
        theme: &cosmic::Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        for ((child, tree), layout) in self
            .children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
        {
            child
                .as_widget()
                .draw(tree, renderer, theme, style, layout, cursor, viewport);
        }
    }

    fn children(&self) -> Vec<Tree> {
        self.children.iter().map(Tree::new).collect()
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(&mut self.children);
    }
}

impl<'a, Msg: 'a> From<WorkspaceStrip<'a, Msg>> for cosmic::Element<'a, Msg> {
    fn from(widget: WorkspaceStrip<'a, Msg>) -> Self {
        cosmic::Element::new(widget)
    }
}
