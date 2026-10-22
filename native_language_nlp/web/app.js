const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const imageInput = document.querySelector("#image-input");
const attachment = document.querySelector("#attachment");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");

let selectedImage = null;

function addMessage(text, role) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}-message`;
  wrapper.innerHTML = `<div class="avatar">${role === "user" ? "You" : "K"}</div><div class="bubble"></div>`;
  const bubble = wrapper.querySelector(".bubble");
  if (role === "bot") {
    bubble.innerHTML = renderMarkdown(text);
    const speakButton = document.createElement("button");
    speakButton.className = "speak-button";
    speakButton.type = "button";
    speakButton.title = "Read response aloud";
    speakButton.textContent = "🔊";
    speakButton.addEventListener("click", () => speakResponse(text, speakButton));
    bubble.appendChild(speakButton);
  } else {
    bubble.textContent = text;
  }

  async function speakResponse(text, button) {
    button.disabled = true;
    button.textContent = "…";
    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error || "Text-to-speech failed.");
      }
      const audioUrl = URL.createObjectURL(await response.blob());
      const audio = new Audio(audioUrl);
      audio.addEventListener("ended", () => URL.revokeObjectURL(audioUrl), { once: true });
      await audio.play();
    } catch (error) {
      addMessage(`Audio error: ${error.message}`, "bot");
    } finally {
      button.disabled = false;
      button.textContent = "🔊";
    }
  }
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inCodeBlock = false;
  let codeLines = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
    } else if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  }

  if (inCodeBlock) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  closeList();
  return html.join("");
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  value = value.replace(/_([^_]+)_/g, "<em>$1</em>");
  return value;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

imageInput.addEventListener("change", () => {
  selectedImage = imageInput.files[0] || null;
  attachment.textContent = selectedImage ? `Attached: ${selectedImage.name}` : "";
  attachment.classList.toggle("hidden", !selectedImage);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message && !selectedImage) return;

  addMessage(message || "Please analyze this plant image.", "user");
  input.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "...";

  try {
    const payload = { message: message || "Please detect the disease in this plant image." };
    if (selectedImage) {
      payload.imageName = selectedImage.name;
      payload.image = await fileToDataUrl(selectedImage);
    }
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The server could not process the request.");
    addMessage(result.reply, "bot");
  } catch (error) {
    addMessage(`Error: ${error.message}`, "bot");
  } finally {
    selectedImage = null;
    imageInput.value = "";
    attachment.classList.add("hidden");
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    input.focus();
  }
});

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}
