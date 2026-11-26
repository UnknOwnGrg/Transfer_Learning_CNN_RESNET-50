// Address of the backend API. Change this if the backend runs elsewhere.
const API_URL = "http://127.0.0.1:8000";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const dropzoneText = document.getElementById("dropzone-text");
const predictBtn = document.getElementById("predict-btn");
const result = document.getElementById("result");
const resultLabel = document.getElementById("result-label");
const resultDetail = document.getElementById("result-detail");
const barFill = document.getElementById("bar-fill");
const errorBox = document.getElementById("error");

let selectedFile = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function selectFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Please choose an image file.");
    return;
  }
  selectedFile = file;
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropzoneText.hidden = true;
  predictBtn.disabled = false;
  result.hidden = true;
  errorBox.hidden = true;
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";
  errorBox.hidden = true;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);

    const isDog = data.label === "dog";
    resultLabel.textContent = `${isDog ? "Dog" : "Cat"} · ${(data.confidence * 100).toFixed(1)}%`;
    resultLabel.className = `result-label ${data.label}`;
    barFill.style.width = `${data.p_dog * 100}%`;
    resultDetail.textContent = `P(dog) = ${data.p_dog.toFixed(3)}`;
    result.hidden = false;
  } catch (err) {
    showError(
      err instanceof TypeError
        ? `Can't reach the backend at ${API_URL}. Is it running?`
        : err.message
    );
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
