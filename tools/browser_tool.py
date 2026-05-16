import os
import time
from playwright.sync_api import sync_playwright

class BrowserTool:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.artifacts_dir = os.path.join(workspace_root, ".999", "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.browser = None
        self.page = None
        self.playwright = None
        
    def _ensure_browser(self):
        """Ensures the browser is launched."""
        if not self.browser:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=False)
            self.page = self.browser.new_page()
            
    def navigate(self, url: str) -> str:
        """Navigates to a URL."""
        try:
            self._ensure_browser()
            self.page.goto(url, timeout=30000)
            return f"Successfully navigated to {url}"
        except Exception as e:
            return f"Failed to navigate: {str(e)}"
            
    def click(self, selector: str) -> str:
        """Clicks an element."""
        try:
            self._ensure_browser()
            self.page.click(selector, timeout=5000)
            return f"Clicked element: {selector}"
        except Exception as e:
            return f"Failed to click: {str(e)}"
            
    def type(self, selector: str, text: str) -> str:
        """Types text into an element."""
        try:
            self._ensure_browser()
            self.page.type(selector, text, timeout=5000)
            return f"Typed text into: {selector}"
        except Exception as e:
            return f"Failed to type: {str(e)}"
            
    def screenshot(self, name: str = "screenshot") -> str:
        """Takes a screenshot and saves it to artifacts."""
        try:
            self._ensure_browser()
            filename = f"{name}_{int(time.time())}.png"
            path = os.path.join(self.artifacts_dir, filename)
            self.page.screenshot(path=path)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Failed to take screenshot: {str(e)}"
            
    def close(self):
        """Closes the browser."""
        if self.browser:
            self.browser.close()
            self.playwright.stop()
            self.browser = None
            self.page = None
            self.playwright = None
