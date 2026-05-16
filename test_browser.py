import os
import sys
# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.browser_tool import BrowserTool

def main():
    print("Initializing BrowserTool...")
    browser = BrowserTool(".")
    
    try:
        print("Navigating to Google...")
        result = browser.navigate("https://www.google.com")
        print(result)
        
        print("Taking screenshot...")
        result = browser.screenshot("google_test")
        print(result)
        
    finally:
        print("Closing browser...")
        browser.close()
        print("Done!")

if __name__ == "__main__":
    main()
