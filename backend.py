import os
import json
import cgi
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class names
model = load_model("disease_prediction_model.h5")
class_names = sorted(os.listdir('PlantVillage'))

class MyHandler(BaseHTTPRequestHandler):
    def _set_headers_json(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _set_headers_html(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self._set_headers_html()
            with open('templates/index.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path.startswith('/static/'):
            filepath = self.path.lstrip('/')
            if os.path.exists(filepath):
                self.send_response(200)
                if filepath.endswith('.css'):
                    self.send_header('Content-type', 'text/css')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Page not found")

    def do_POST(self):
        if self.path == '/predict':
            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
            if ctype == 'multipart/form-data':
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                pdict['CONTENT-LENGTH'] = int(self.headers['content-length'])
                fields = cgi.parse_multipart(self.rfile, pdict)
                file_data = fields.get('image')[0]

                # Load and preprocess image
                img = image.load_img(BytesIO(file_data), target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]

                # Respond
                self._set_headers_json()
                response = {'prediction': predicted_class}
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(400, "Invalid content type")

def run(server_class=HTTPServer, handler_class=MyHandler, port=5000):
    print(f"🚀 Server running at http://localhost:{port}")
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

if __name__ == '__main__':
    run()
