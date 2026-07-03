# How to record the demo GIF

The README embeds `docs/demo.gif`. Replace it with a real recording of the live
dashboard so the project shows something the moment a recruiter opens it.

## Tool (Windows)

Use **[ScreenToGif](https://www.screentogif.com/)** — free, records a screen region
straight to an optimized `.gif`.

## What to capture (~15–25 seconds, keep it tight)

Open the live Space (or run locally) and record this flow:

1. The **Live Operations Console** with the real-time transaction stream running.
2. Toggle **"Attack Mode"** so a burst of high-risk transactions lights up the alert queue.
3. Click one alert to open the **investigation panel** and show the **SHAP risk drivers**.
4. (Optional) The **CSV Batch Processor** scoring an uploaded file.

## Keep the file small

- Target **under ~8 MB** so it loads fast on GitHub (crop to the app, ~12–15 fps).
- In ScreenToGif use *File → Save as → GIF* and enable the built-in optimizer, or
  export MP4 and convert. Save it as **`docs/demo.gif`** (exact path/name).

## Then

```bash
git add docs/demo.gif && git commit -m "docs: add live demo GIF" && git push
```

Also update the **▶️ Try it live** link at the top of the README with your real
Hugging Face Space URL.
