import screeninfo


def get_dynamic_dimensions():
    """Get dynamic dimensions based on screen size."""
    screen = screeninfo.get_monitors()[0]  # Get primary screen dimensions
    width = int(screen.width)  # Use 80% of the screen width
    height = int(screen.height)  # Use 60% of the screen height

    return width, height


WIDTH, HEIGHT = get_dynamic_dimensions()
print(f'{WIDTH}x{HEIGHT}')

SIZING_MODE='scale_both'
