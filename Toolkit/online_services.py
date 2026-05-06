import os
import json
import base64
import time
import httpx
from google import genai
from google.genai import types
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
        self._genai_client = None

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        success_count = 0
        total = len(self.items)
        clients = {}

        if "gemini" in self.service_type:
            api_key = self.config.get('gemini_key', '')
            if api_key:
                self._genai_client = genai.Client(api_key=api_key)

        try:
            for i, item in enumerate(self.items):
                if self._is_cancelled:
                    break

                try:
                    msg = f"[{i+1}/{total}] Processing {item.filename}..."
                    self.log.emit(msg)
                    print(msg)
                    output = ""

                    if self.service_type.startswith("gemini"):
                        prompt_key = 'tags_prompt' if self.service_type == "gemini_tags" else 'caption_prompt'
                        prompt = self.config.get(prompt_key, 'Describe this image.')
                        output = self.call_gemini_sdk(item.image_path, prompt)
                    
                        if i < total - 1:
                             time.sleep(4) 
                             
                    else:
                        # Gradio Services
                        space_id = ""
                        args = []
                        api_name = None
                        
                        if self.service_type == "wd_tagger":
                            space_id = "SmilingWolf/wd-tagger"
                            args = ["SmilingWolf/wd-swinv2-tagger-v3", 0.35, False, 0.5, False]
                            api_name = "/predict"
                        elif self.service_type == "joy_tag":
                            space_id = "fancyfeast/joytag"
                            api_name = "/predict"
                        elif self.service_type == "joy_caption":
                            space_id = "fancyfeast/joy-caption-beta-one"
                            args = [self.config.get('caption_prompt', 'Write a long detailed description for this image.')]
                            api_name = "/chat_joycaption"

                        if space_id not in clients:
                            msg = f"Connecting to HF Space: {space_id}..."
                            self.log.emit(msg); print(msg)
                            clients[space_id] = Client(space_id)
                        
                        client = clients[space_id]
                        if api_name:
                            result = client.predict(handle_file(item.image_path), *args, api_name=api_name)
                        else:
                            result = client.predict(handle_file(item.image_path), *args)
                        
                        if isinstance(result, (list, tuple)):
                            result = result[0]
                        output = str(result)

                    if output and output.strip() and output.lower() != "none":
                        clean_output = self.clean_output(output)
                        item.save_caption(clean_output)
                        success_count += 1
                        msg = f"✓ Tagged {item.filename}"
                        self.log.emit(msg); print(msg)
                    else:
                        msg = f"⚠ Empty output for {item.filename}"
                        self.log.emit(msg); print(msg)

                except Exception as e:
                    msg = f"✗ Error {item.filename}: {str(e)}"
                    self.log.emit(msg); print(msg)

                self.progress.emit(int((i + 1) / total * 100))

        except Exception as e:
            self.log.emit(f"Critical Worker Error: {str(e)}")
        finally:
            for c in clients.values():
                try: c.close()
                except: pass

        self.finished.emit(success_count)

    def call_gemini_sdk(self, image_path, prompt):
        if not self._genai_client:
            raise Exception("Gemini API Client not initialized. Check API Key.")

        model_id = self.config.get('gemini_model', 'gemini-2.0-flash')
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
#rety ratelimit
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._genai_client.models.generate_content(
                    model=model_id,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/webp")
                    ]
                )
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 30 + (attempt * 30)
                    self.log.emit(f"⚠ Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise e

    def clean_output(self, text):
        text = text.replace("Caption:", "").replace("Tags:", "").strip()
        # Remove backticks and markdown code blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[-2].strip()
                if text.startswith("json") or text.startswith("text"):
                    text = "\n".join(text.split("\n")[1:])
        return text.strip()

