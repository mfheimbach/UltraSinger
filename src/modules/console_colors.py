"""Colors for the console"""

ULTRASINGER_HEAD = "\033[92m[UltraSinger]\033[0m"


def blue_highlighted(text: str) -> str:
    """Returns a blue highlighted text"""
    return f"{Bcolors.blue}{text}{Bcolors.endc}"


def green_highlighted(text: str) -> str:
    """Returns a blue highlighted text"""
    return f"{Bcolors.dark_green}{text}{Bcolors.endc}"


def red_highlighted(text: str) -> str:
    """Returns a red highlighted text"""
    return f"{Bcolors.red}{text}{Bcolors.endc}"


class Bcolors:
    """Colors for the console"""

    blue = "\033[94m"
    dark_green = "\033[32m"
    red = "\033[91m"
    light_blue = "\033[96m"
    cyan = "\033[36m"
    gold = "\033[93m"
    underline = "\033[4m"
    endc = "\033[0m"
    bright_green = "\033[38;2;204;255;204m"
