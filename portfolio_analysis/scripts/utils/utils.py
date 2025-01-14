def colorize(val):
    try:
        number = int(val)
    except (ValueError, TypeError):
        return ""

    # Start with bold formatting for all numeric cells
    style = "font-weight: bold; "

    # Append color based on positive or negative
    if number > 0:
        style += "color: green;"
    elif number < 0:
        style += "color: red;"

    return style

