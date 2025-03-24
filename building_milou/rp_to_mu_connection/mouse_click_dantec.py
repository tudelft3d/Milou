import pyautogui
import time

pyautogui.PAUSE = 0.5
pyautogui.useImageNotFoundException()


def test_mouse():
    pyautogui.moveTo(1750, 30, duration=0.5)
    pyautogui.click()
    # pyautogui.rightClick()


def comfortSense():
    while True:
        try:
            time.sleep(1)
            # Locate the center of the button
            x_location, y_location = pyautogui.locateCenterOnScreen('start_measurement_button.PNG')

            # Click the button
            pyautogui.click(x_location, y_location)
            pyautogui.moveTo(5, 5, duration=0.5)
            break
        except pyautogui.ImageNotFoundException:
            pass

    while True:
        try:
            time.sleep(1)
            pyautogui.locateOnScreen('start_measurement_button.PNG')
            break
        except pyautogui.ImageNotFoundException:
            pass


if __name__ == '__main__':
    # test_mouse()  # this is to minimise the pycharm window
    comfortSense()

