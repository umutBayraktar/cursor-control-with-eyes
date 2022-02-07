import pyautogui


if __name__ == '__main__':
    screen_sizes = pyautogui.size()
    height = screen_sizes.height
    width = screen_sizes.width
    print(height)
    print(width)
    half_width = int(width / 2)
    pyautogui.moveTo(half_width, 0)
    pyautogui.moveTo(half_width, height, 5)