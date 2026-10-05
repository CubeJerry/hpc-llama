"""Whole-cell scrollbars for terminals with inconsistent block-glyph rendering."""
from rich.color import Color
from rich.segment import Segment, Segments
from rich.style import Style
from textual.scrollbar import ScrollBarRender


class PortableScrollBarRender(ScrollBarRender):
    """Keep Textual's mouse actions, using explicit backgrounds and no reverse."""

    @classmethod
    def render_bar(
        cls, size=25, virtual_size=50, window_size=20, position=0,
        thickness=1, vertical=True, back_color=Color.parse("#555555"),
        bar_color=Color.parse("bright_magenta"), *, monochrome=False,
    ) -> Segments:
        size = max(0, int(size))
        thickness = max(1, int(thickness))
        width = thickness if vertical else 1
        start = end = 0
        if size and 0 < window_size < virtual_size:
            length = min(size, max(1, round(size * window_size / virtual_size)))
            ratio = min(1, max(0, position / (virtual_size - window_size)))
            start = round((size - length) * ratio)
            end = start + length
        segments = []
        for index in range(size):
            thumb = start <= index < end
            action = "grab" if thumb else "scroll_up" if index < start else "scroll_down"
            style = Style(bgcolor=bar_color if thumb else back_color,
                          meta={"@mouse.down": action} if end else None)
            segments.append(Segment(("#" if monochrome and thumb else " ") * width, style))
        if vertical:
            return Segments(segments, new_lines=True)
        return Segments((segments + [Segment.line()]) * thickness, new_lines=False)

    def __rich_console__(self, console, options):
        style = console.get_style(self.style)
        yield self.render_bar(
            size=(options.height or console.height) if self.vertical else (options.max_width or console.width),
            window_size=self.window_size, virtual_size=self.virtual_size,
            position=self.position, vertical=self.vertical,
            thickness=(options.max_width or console.width) if self.vertical else (options.height or console.height),
            back_color=style.bgcolor or Color.parse("#555555"),
            bar_color=style.color or Color.parse("bright_magenta"),
            monochrome=console.color_system is None or console.no_color,
        )
