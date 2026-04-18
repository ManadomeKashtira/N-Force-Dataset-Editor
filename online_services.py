import os
import json
import base64
import httpx
from gradio_client import Client, handle_file
from PyQt6.QtCore import QThread, pyqtSignal

class OnlineWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, items, service_type, config):
        super().__init__()
        self.items = items
        self.service_type = service_type
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        success_count = 0
        total = len(self.items)

        for i, item in enumerate(self.items):
            if self._is_cancelled:
                break

            try:
                self.log.emit(f"Processing {item.filename}...")
                output = ""

                if self.service_type == "gemini_tags":
                    output = self.call_gemini(item.image_path, self.config.get('tags_prompt', ''))
                elif self.service_type == "gemini_caption":
                    output = self.call_gemini(item.image_path, self.config.get('caption_prompt', ''))
                elif self.service_type == "wd_tagger":
                    output = self.call_hf_space("SmilingWolf/wd-tagger", item.image_path, 0.35)
                elif self.service_type == "joy_tag":
                    output = self.call_hf_space("fancyfeast/joytag", item.image_path)
                elif self.service_type == "joy_caption":
                    output = self.call_hf_space("fancyfeast/joy-caption-beta-one", item.image_path, self.config.get('caption_prompt', ''))

                if output:
                    clean_output = self.clean_output(output)
                    item.save_caption(clean_output)
                    success_count += 1

            except Exception as e:
                self.log.emit(f"Error {item.filename}: {str(e)}")

            self.progress.emit(int((i + 1) / total * 100))

        self.finished.emit(success_count)

    def call_gemini(self, image_path, prompt):
        api_key = self.config.get('gemini_key', '')
        if not api_key:
            raise Exception("Gemini API Key missing")

        model_id = self.config.get('gemini_model', 'gemini-1.5-flash')
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
                ]
            }]
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']

    def call_hf_space(self, space_id, image_path, *args):
        client = Client(space_id)

        result = client.predict(handle_file(image_path), *args)
        
        if isinstance(result, tuple):
            result = result[0]
        
        if isinstance(result, dict):

            tags = [k for k, v in result.items() if isinstance(v, (int, float)) and v > 0.3]
            return ", ".join(tags)
        
        return str(result)

    def clean_output(self, text):

        text = text.replace("Caption:", "").replace("Tags:", "").strip()
        # Remove backticks and markdown code blocks
        if "```" in text:
            text = text.split("```")[-2].strip()
            if text.startswith("json") or text.startswith("text"):
                text = "\n".join(text.split("\n")[1:])
        return text.strip()
