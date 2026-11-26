# Frontend: Cat or Dog?

A plain HTML/CSS/JavaScript page. You pick or drag in a photo, it is sent to the backend, and the page shows "Cat" or "Dog" with the confidence and a probability bar. There is no build step and nothing to install.

## Run

1. Start the backend first (see `../backend/README.md`). It must be running on http://127.0.0.1:8000.
2. Serve this folder on port 5500:

   ```bash
   cd frontend
   python -m http.server 5500
   ```

3. Open http://127.0.0.1:5500 in your browser.

Use a local server like this rather than double-clicking `index.html`. Browsers block requests from `file://` pages to the API (CORS). VS Code's Live Server extension also works, since it uses port 5500 by default.

## Configuration

The backend address is set at the top of `app.js`:

```js
const API_URL = "http://127.0.0.1:8000";
```

If you serve the frontend on a different port, add that address to the backend's `ALLOWED_ORIGINS`.

## Files

| File         | Purpose                                     |
|--------------|---------------------------------------------|
| `index.html` | Page layout                                 |
| `style.css`  | Styles (follows the system light/dark theme)|
| `app.js`     | File selection, upload and showing results  |
