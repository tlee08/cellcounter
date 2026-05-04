def confirm(prompt: str, *, default: bool = False) -> bool:
    """Get yes/no confirmation from user."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {hint}: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")
