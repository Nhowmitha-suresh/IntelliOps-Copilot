# Scratch AI Studio

Build a dark-themed, modern, animated web app to demo a from-scratch transformer language model I trained. This is a portfolio/technical showcase project — the UI should feel premium, professional, and slightly futuristic, like an AI research lab's internal tool.

BACKEND CONTEXT (I'll connect this myself later, just build the UI with mock/placeholder data for now):

- POST /generate — body: { prompt, model: "baseline" | "variant", max_new_tokens, temperature, repetition_penalty }, returns { generated_text }

- GET /model-info — returns { baseline: { param_count, vocab_size, val_perplexity }, variant: { param_count, vocab_size, val_perplexity } }

VISUAL DESIGN:

- Dark background (near-black or deep navy) with a subtle animated gradient mesh or slowly drifting particle/node network in the background — should feel like a neural network, calm and ambient, not distracting

- Accent colors: teal-to-purple gradient for primary actions and highlights

- Clean, modern sans-serif typography, generous whitespace

- Hero section at the top: project title "MiniGPT-Scratch" with tagline "A transformer language model built and trained entirely from scratch — no pretrained weights, no APIs"

CORE LAYOUT & FEATURES:

1. MODEL SELECTOR (the centerpiece, not hidden in settings):

   - Two large glowing cards side by side: "Baseline" and "Variant"

   - Each card shows: parameter count, vocab size, validation perplexity, styled like a spec sheet

   - Clicking a card selects it (visually highlight the active one with a glow/border animation)

   - Optionally include a "Compare Both" mode toggle above the cards

2. PROMPT INPUT SECTION:

   - Large textarea for the prompt

   - Row of clickable example prompt chips below it (e.g. "Once upon a time", "To be or not to be", "The King said")

   - A prominent gradient "Generate" button with a hover glow effect

3. ADVANCED SETTINGS (collapsible accordion, closed by default):

   - Temperature slider (0.1–1.5) with live numeric display

   - Max tokens slider (10–500)

   - Repetition penalty slider (1.0–2.0)

4. OUTPUT AREA:

   - Generated text appears with a typewriter/character-reveal animation

   - A subtle pulsing loading animation while "generating" (animated dots or a pulsing node graphic, not a generic spinner)

   - Copy-to-clipboard icon button on the output

   - If "Compare Both" mode is active, show two output panels side by side, one per model, generating simultaneously

5. GENERATION HISTORY:

   - A collapsible sidebar or bottom panel listing the last 5-10 generations from this session, each tagged with which model produced it and a timestamp

   - Clicking a history item re-displays it in the output area

6. "HOW IT WORKS" SECTION (below the main tool, collapsible):

   - A brief, friendly explanation of tokenization → transformer → generation, written for a non-technical visitor

TECHNICAL:

- React with Tailwind CSS and shadcn/ui components

- Use Framer Motion or CSS animations for the reveal/glow/hover effects

- For now, since there's no live backend connected, use realistic mock responses when "Generate" is clicked, with a fake ~1-2 second delay to simulate the real API call, and clearly structure the fetch call so I can swap in the real backend URL later

- Fully responsive for mobile and desktop

Make this feel polished and impressive — this is the frontend for my final-year AI/ML capstone project.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/59216094-2d51-4c37-94c8-b48702047961).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
