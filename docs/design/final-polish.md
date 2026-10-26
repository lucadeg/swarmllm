# Final polish: one ordered fix list

Built from five studies (consistency, first-timer, developer, mobile, designer) of the landing page and /room.
Scope is polish only. Duplicates are merged. Where studies disagree, the rule is clarity first, then consistency.

The two sets touch different files:
- **LANDING** owns `index.html` and `site/**`, including the new `site/css/tokens.css`.
- **ROOM** owns `p2p.html`, `room.js` and `room/**`. It only *links* `site/css/tokens.css` and never edits it.

ROOM R1 depends on LANDING L1 because it needs `site/css/tokens.css` to exist. Until then, the room keeps its own `:root` values, set to the numbers below.

## Shared decisions (both sets use these exact values)

**Tokens.** The canonical names are the landing names:
`--bg --bg-2 --bg-3 --white --ink --ink-2 --ink-3 --ink-4 --rule --rule-2 --rule-3 --blue --blue-text`

**Blue ramp** (replaces the roughly 14 ad hoc blues):

| Token | Value |
|---|---|
| `--blue-50` | `#EAEDFE` (same as `--blue-soft`) |
| `--blue-200` | `#B9C6FF` |
| `--blue-400` | `#7C8FFF` |
| `--blue-500` | `#2A45E0` (same as `--blue`) |
| `--blue-600` | `#2239C8` (same as `--blue-text`, the hover color) |
| `--blue-700` | `#1C33B8` |

Every glow, hot cell, dark-screen tint, `SWATCH` and `PACKET` color uses one of these steps.

**Device colors.** `--dev: #2A45E0, #2B2F3C, #7C8FFF, #5E616B, #B9C6FF, #1C33B8`, exposed as `--d0`…`--d5`.
- The dark neutrals sit between similar blues, so devices 1 and 4 no longer collide.
- A device that only asks uses `--ink-4`.
- The designer's all-blue alternative was rejected: four or more devices become hard to tell apart.

**Radius:** `--r-xs 4` (cells, inline code, status chips), `--r-sm 8` (controls, buttons), `--r-md 12` (rows, tool cards, popovers, composer), `--r-lg 16` (panels, sheets, windows, bubbles), `--r-pill 999`.

**Type:**
- Text sizes `--t-xs 11`, `--t-sm 12`, `--t-md 13`, `--t-base 14`, `--t-lg 15`, `--t-xl 17`. Mono labels use only 11 or 12.
- Display sizes `--d-sm 22`, `--d-md 26`, `--d-lg 36`, `--d-xl clamp(38px,5vw,60px)`.
- No half-pixel sizes.

**Diff and agent UI:**
- `--diff-add #E9F4EC`, `--diff-del #FBEAE8`, `--diff-del-ink #6B1D16`, `--del #B42318` (room `--err` becomes an alias).
- Tool rows: 12px mono, 32px tall.
- Status chips: 11px mono, radius 4.
- Bubbles: 15px text, radius 16.

**Icons:** stroke 1.4 at 16px and up, 1.3 at 12 to 14px.

**Buttons:**
- Blue is the only primary.
- Black is only for Stop and destructive-neutral actions.
- The secondary button is white with a 1px `--rule-3` line.

**Copy:**
- Sentence case for every visible label.
- Mono only for codes, paths, ports and numbers.
- No em dashes.

**Model names:**
- In UI, one scheme with no space after "Qwen": `Qwen3 1.7B`, `Qwen3.8 27B`, `Qwen3.6 35B MoE`.
- In landing captions: "a big open model (35B)".

## LANDING (index.html, site/*)

1. **Shared tokens.** Create `site/css/tokens.css` holding the `:root` block from the shared decisions: colors, blue ramp, `--d0..--d5`, radius, type and diff tokens.
   - Load it in `index.html` before `site.css`.
   - Delete the duplicated values from `site.css :root`.
   - Replace the stray blues `#8FA2FF #5B74FF #93A6FF #C8D0FA` with the nearest ramp step, and keep `#7C8FFF` and `#1C33B8` as tokens.
2. **Stop the auto-scroll.** Delete the "first load: glide the demo to the middle" IIFE in `site/js/demo.js` (about lines 568 to 597). The `hookHold`/`hook` logic falls back to no hold.
   - Fit the hero and the demo's top edge into 1280x720 with no scroll: set `--body-h:clamp(380px,calc(100svh - 420px),580px)` and reduce the `.hero` top and bottom padding by about 30%.
3. **Opening frame preview width.** In `site.css`, add `.demo.hook .pv{width:100%}`. This removes the roughly 260px empty strip in the opening Tetris frame.
4. **Scene crossfade.** In `.js .scene`, set the incoming scene's `transition-delay` to `.45s` (the outgoing duration) instead of `.15s`, so two scenes are never visible at once.
5. **Step 5 caption.** In `index.html` (`.sb-dots [data-step=4] .lbl`) and in `demo.js` if it duplicates the text, change it to "Every word passes through every device, 40 layers in total."
   - Rename the "hidden state 10 KB per hop" label to "about 10 KB between devices per word".
   - In captions, say "a big open model (35B)".
6. **Step timing and caption height.**
   - In `demo.js`, give every step at least 2.5s. Step 1 goes from about 1s to 2.5s, and steps 2 and 4 go to 2.5s. Cut step 5 from about 8s to about 5s.
   - Give the caption element `min-height` for two lines so the content below stops shifting about 8px on each step.
7. **Believable download ETA.** In the demo's step 4, change "about 20 s left" to "about 4 min left", with numbers that match about 100 MB/s.
8. **Demo join step matches the real join screen.**
   - Swap `#i-pen` for the room's shuffle glyph (add it as a sprite symbol).
   - Change the copy to the real one ("…to type on your other devices").
   - Dim the second tab (Desktop) to `opacity:.4` until step 2.
9. **Phone truncation in the demo.** At 640px and below, `.hl`/`.hg`:
   - Stack the device name above "layers 31-40".
   - Drop ", ready" and ", from cache" and keep only the GB figure ("5.6 GB").
   - Shorten "3.9 of 5.6 GB" to "3.9/5.6 GB".
10. **Headline and CTA copy.**
    - Change "Start a free room" to "Start a room" (nav and closer).
    - Change the hero `.sub` and `<meta name=description>` from "One laptop can't hold a big open AI model." to "A model too big for any one of your devices runs across all of them." Keep the rest of the sentence.
    - Closer CTA: replace `#i-arrow` (the outward ↗) with a right arrow `#i-next` (new sprite symbol, stroke 1.4) that nudges 2px on hover. Keep ↗ for GitHub.
    - Keep "Star on GitHub" as the second button.
11. **One requirements line.** Add one quiet line under the demo, 13px `--ink-3`, centred: "Chrome or Edge on a laptop or desktop · first run downloads a few GB per device, cached after · phones can join and chat". No new section.
12. **Social preview tags.** Add `og:title`, `og:description`, `og:url`, `og:image`, `twitter:card=summary_large_image` and `twitter:image` to `index.html`.
    - Point the image at `/site/og.png`, a 1200x630 still of the opening Tetris frame taken with Playwright and saved in `site/`.
13. **Mini app palette.** In `site/js/apps.js`, recolor the Breakout bricks, and any other rainbow palettes in the mini apps, to the blue ramp (`--blue-200/400/500/700`) plus one warm accent (`#E08A2A`) for the ball or score. Tetris already does this.
14. **Empty demo frames.** The chat scene (`.s-chat .msgs`), the picker (`.pick`) and the split (`.model`) sit low in a tall empty stage.
    - Centre their content vertically with `justify-content:center` and a top padding of `max(24px, 8%)`.
    - Keep the demo window's fixed height, since the mobile study praised it.
15. **Agent UI values match the room.** Set these landing values to the shared numbers:
    - `.diff .del/.add` use the diff tokens.
    - `.tl .st` goes to 11px, radius 4.
    - `.tool` goes to 12px mono, 32px tall.
    - `.q p` bubbles go to 15px, radius 16.
    - `.composer` goes to radius 12.
16. **Radius and type scale.**
    - Snap the landing radii 13 and 15 to 12 and 16, and the other off-scale radii to the nearest step.
    - Snap 10.5, 11.5, 12.5 and 13.5 to the type steps: labels go down and body text goes up.
    - Use the `--t-*` and `--d-*` tokens.
17. **Icon strokes.** In the `index.html` sprite:
    - file and pen: 1.1 to 1.3.
    - lock and reload: 1.2 to 1.3.
    - laptop: 1.3.
    - down and arrow: 1.4.
    - check: 1.6 to 1.4.
18. **Realistic line counts** in the demo file tree and in the no-JS fallback: `index.html` about 24, `style.css` about 60, `game.js` about 210. The HTML and the JS must show the same numbers.
19. **Model names.** Rename `.pk-n` and `.who` "Qwen 3.8 27B" and "Qwen 3.6 35B MoE" to "Qwen3.8 27B" and "Qwen3.6 35B MoE", in `index.html` and `demo.js`.
20. **Tap targets.** Under `@media (pointer:coarse)`:
    - The nav CTA, the `.modes` tabs, the step dots and `#replay` get at least 44px of hit area, using `::before{content:"";position:absolute;inset:-8px}`.
    - Give each of them `position:relative`.
21. **Closing logo gather.** In `site/js/swarm.js` (`#swarm2`):
    - Ease the gather with `cubic-bezier(.2,.7,.2,1)` so it is mostly formed by 40%.
    - Start it when the section is 30% visible.

## ROOM (p2p.html, room.js, room/*)

1. **Link the shared tokens.**
   - Add `<link rel="stylesheet" href="/site/css/tokens.css">` before the inline `<style>` in `p2p.html`.
   - In the room's `:root`, turn the room names into aliases: `--text:var(--ink)`, `--border:var(--rule)`, `--border-2:var(--rule-3)`, `--accent:var(--blue)`, `--accent-text:var(--blue-text)`, `--accent-hover:var(--blue-600)`, `--panel-3:var(--bg-2)`, `--panel-2:var(--bg-3)`, `--err:var(--del)`, `--diff-del:var(--diff-del)`.
   - Delete the unused tokens `--user-bubble`, `--accent-dim` and `--lift`.
   - Replace the roughly 12 literal `#2239C8` with `var(--accent-hover)`.
   - Map `#8EA2FF #9AA8F0 #4A5FD0 #6E86FF #B9C6FF #AEB7E5 #C3CCF8 #1F2FA8` to ramp steps. `room/compute.js PACKET` becomes `[--blue-500, --blue-400, --blue-200]`.
2. **One color per device, everywhere.**
   - `room.js SWATCH = ["#2A45E0","#2B2F3C","#7C8FFF","#5E616B","#B9C6FF","#1C33B8"]`.
   - Assign the color once per device, keyed by peer id in join order, and store it on the peer record.
   - Use that one lookup in the header chips, the pool bar, the loading rows, the band lanes and bars, and the Lend screen.
   - Devices that only ask get `--ink-4` on every screen, including their own.
3. **Lend button in the header.**
   - Replace `#compute-open .mk` (the dots logo) with a chip glyph: a 12px rounded square with 2 pins per side, stroke 1.4.
   - Keep `.lbl` "Lend" at 820px and wider.
   - Use the same glyph for the Lend button in the chip popover.
   - The animated dots mark stays only on the Lend screen.
4. **Chip popover invisible on phones.**
   - Remove `mask-image` from `body.in-room #peers` (p2p.html about line 1026).
   - Draw the fade with `::after` on `#peers`' parent instead: `position:absolute; right:0; width:40px; background:linear-gradient(90deg,transparent,var(--bg)); pointer-events:none`.
   - Give `#hdr-sum` a solid `--bg` background with a 24px left gradient so "32 GB pooled" never sits on top of a chip.
5. **Tooltips stick after a tap.**
   - Change `[data-tip]:is(:hover,:focus-visible)::after` (about line 170) to `[data-tip]:focus-visible::after` plus `@media (hover:hover){ [data-tip]:hover::after{…} }`.
   - Hide the tooltip on `pointerdown`.
6. **One primary button.**
   - `#code-pane #ed-save`, `.cm-approve button.ok` and `.cm-jump` get the `button.primary` look: blue fill, `--accent-hover` on hover, radius 8.
   - `.compute-btn` ("Lend this device" in the device card) becomes secondary (white, `--rule-3` line).
   - Black stays only on Stop.
7. **Radius scale.**
   - Buttons: `button`, `.hbtn`, `.mbtn`, `#code-pane button`, `#chat-tools button`, `#ai-start` go to 8. `.cm-jump` and `.seg.chips` go to 999.
   - Cards: `.pop` and `.toast` go to 12. `.jpanel`, `#ai-panel`, `.sheet` and `.stepbar` go to 16.
   - `#ai-row` and `#code-row` go from 11 to 12.
   - `.m .bubble` goes to radius 16 and 15px text.
   - `.chip` (status) goes to 11px mono, radius 4.
   - `.cm-tool` goes to 12px mono, 32px tall.
8. **Type scale.**
   - Snap the room's 21 sizes to the tokens.
   - 10 and 10.5 become 11.
   - 11.5, 12 and 12.5 become 12, including `.ap-note`, `.rung .nd` and `.rung b`.
   - 13.5 becomes 13 (`.ap-h span`).
   - 14.5 becomes 15.
   - 15.5 and 16 become 15, except iOS input fields, which stay 16.
   - 18 becomes 17.
   - 20 becomes 22.
   - 23 and 24 become 22 or 26.
9. **Sentence case for labels** (keep the ids).
   - `#new-chat` "New chat", `#continue-btn` "Continue", `#regen-btn` "Regenerate".
   - `#code-newtask` "New task", `#code-auto-l` "Auto-approve edits".
   - `#card-save` "Save PNG", `#card-share` "Share", `#card-close` "Close".
   - `#pv-con-toggle` "Console", `#pv-clear` "Clear", `#pv-to-agent` "Send to agent", `#pv-open` "Open".
   - `#code-driver` becomes "The host is driving the agent. You see what it does, live." and "click to run" becomes "Click to run".
   - Empty states: "No projects yet", "No files yet", "Nothing served yet" and "No port served", set in Geist, not mono.
   - Section labels `.ms > h3` become sentence case, 13px Geist 500 in `--ink-3`, with no uppercase or tracking.
   - Any test that matches on these strings must be updated along with them.
10. **Text characters to SVG icons.**
    - `#pv-reload` and `#regen-btn` get a reload SVG, a copy of the landing's `#i-reload`, drawn inline.
    - `#pv-open` and `#continue-btn` get the arrow SVG.
    - `#pv-con-toggle` gets a chevron SVG that rotates when open.
    - `.pv-x` gets the room's close SVG.
    - Strokes: 1.4 at 16px, 1.3 at 12 to 14px. Also fix the Invite plus and close (1.6 to 1.4), settings (1.5 to 1.4) and the `--i-file`/`--i-dir` masks (1.1 to 1.3).
11. **Sheets built the same way.** `#share`, `#card` and `#room-over` all get:
    - Header: Funnel `h2` at 26px, one help line at 14px `--ink-2`, and a close ✕ button top right at 12/12.
    - Main action as `button.primary`: "Save PNG" on the card, "Start a new room" on Room over.
    - Escape and a backdrop click close `#share` and the full-size QR, and focus returns to `#share-btn`. Extend the Escape handler at room.js about line 197.
    - On `#share`, add the help line "On your phone? Scan with the camera app."
12. **Tabs.**
    - Restyle `#code-out-tabs` (Preview | Editor) as a small segmented control: 3px well in `--bg-3`, 28px buttons, radius 8, and a white raised selected button like `#mode-bar`.
    - Give `.pv-tab` the same look.
    - Settings "Answer style" pills (`#ai-persona`) use the same selected state: white fill, subtle shadow, ink text, instead of a blue outline.
13. **Code mode vertical space.**
    - In `.code-mode`, `#swarm-map` starts folded to its one-line header on every screen size, reusing the `#band-toggle` state.
    - At 820px and wider, move `#mode-bar` (same node and id) into the header next to `.hd-code` and hide the `.mode-row` strip.
    - Phones keep `.mode-row`.
    - Hide `#mode-bar` until a model is ready, as the sim's `ready()` already does.
14. **Phone chat layout.**
    - Fold the band by default on phones in Chat too. Generalise room.js about line 1969 to `innerWidth < 820`, whatever the mode.
    - `.mode-row`: solid `--bg` background, a 1px `--rule-2` bottom line and 8px below it, so no bubble slides under the pill.
    - `#ai-output`: follow the newest message on `visualViewport` resize and on a `ResizeObserver`, but only if the reader was already within 40px of the bottom.
    - Add `body.kbd` while `#ai-prompt` or `#code-prompt` has focus and `visualViewport.height < 600` on a coarse pointer. It hides `#peers`, `#swarm-map` and `.mode-row`.
15. **Mobile Code mode.**
    - Under 820px, `#code-log` gets `max-height:none; overflow:visible`, so only `#code-pane` scrolls.
    - When a serve succeeds on a phone, scroll the preview into view.
    - `#code-proj-select`: Geist instead of mono, 16px only under `(pointer:coarse)`.
    - `#code-tree .f[data-ext]` icons use `--ink-4`, and the open file gets a 5px `--blue` dot. This drops the off-palette `#B0520A` and `#0E6E8C`.
    - `#code-log` gets `mask-image:linear-gradient(transparent,#000 24px)`.
16. **Code mode empty preview.** Replace the big "localhost" empty state with:
    - A quiet dotted background, the small dots mark, "Your app shows up here", and "The agent serves it on a port like `:5173`", with the port in mono.
    - In `#pv-bar`, a small mono tag "sandbox" next to the address. Tooltip: "Runs in a sandboxed frame in this tab."
17. **Declined edits and long diff lines.**
    - When a `.cm-tool` card is declined: `.cm-diff` gets `opacity:.45` and `text-decoration:line-through`, and a line reads "Declined: <reason>" if one was given.
    - `.cm-diff` rows: `white-space:pre-wrap` with a hanging indent under 640px, and a right-edge fade at wider sizes.
    - The collapsed console shows the error count in `--del` inside `#pv-counts`.
18. **Join screen header matches the landing.**
    - `p2p.html header` on the join screen: 64px tall, content in a `max-width:1240px; margin:0 auto; padding:0 var(--gutter)` wrapper.
    - `.logo a` wordmark at 17px.
    - `#join-screen h1 span` in `var(--accent)` instead of `--muted`.
19. **Join card.**
    - Column titles become "New room" and "Have a code?". The button stays "Start a room".
    - `#join-pledge`: drop the separate borders on `.step` and the input, and wrap them in the `.ap-step` well with "GB" inside. Keep `#join-gb` and the other ids.
    - When preflight fails, replace the stepper visually with "Joins to ask" at 13px `--ink-3`.
    - `#join-btn`: disabled look until `#code-input` has 4 valid characters, then `button.primary`.
    - Add `enterkeyhint="go"` to `#code-input`.
    - `#jw-t`: wrap the code in `<b class="mono">` with `font:500 .9em var(--mono); letter-spacing:.12em`.
20. **WebGPU message.**
    - In `room/preflight.js`, check iPhone, iPad and Android before the `hasGpuApi` branch.
    - Phone wording: "This phone's GPU isn't available to the browser. It can still join and chat."
    - Desktop wording: "This browser can't use its graphics chip, so this device can chat but not help run the model. Chrome or Edge on a laptop works best."
    - Replace the amber box `#join-status.warn`, and the grey footnote that repeats it, with one left-aligned line: an amber dot, the text, and a "Details" `<details>` that holds the `chrome://flags` hint.
    - Use the same wording in `#ap-no`.
21. **Model picker rows and disabled Start.**
    - `#ai-ladder` rows show one right-aligned figure: "4 GB" in mono with the blue "fits" tag when the model fits.
    - When it doesn't fit: "needs 4 GB more" in `--warn-text`, row opacity .6, tooltip "Invite a device to fit this".
    - When the pool is 0 GB, show one line "Needs a device with WebGPU" instead of repeating "short" on every row.
    - When `#ai-start` is disabled, a 13px line under it reads "Add a laptop or desktop to start." and a link "Invite" opens `#share`.
    - In the lobby, move `#ai-panel` up to about 20vh from the top instead of centring it vertically.
22. **Plain words.**
    - Self chip "asks" becomes "chat only", with the tooltip "This device can ask but can't hold model layers".
    - Band stat "ms a lap" becomes "ms per word", with a tooltip for the technical meaning.
    - Settings "Think before answering (Qwen)" becomes "Think before answering (slower, more careful)".
    - Loading card: hide the "40 layers · 3 devices" mono line, which repeats the rows below it (keep the element). Put the "Downloading" pill and the "1.1 GB of 3.1 GB" figure on one baseline, both 12px mono.
23. **Lend screen (`room/compute.js`).**
    - Swap `meta[name=theme-color]` to the dark background in `open()` and restore `#F6F5F1` in `close()`.
    - Stats: `align-items:end` on the cells, and shorten the label to "tokens here".
    - Add one explainer line under the title: "This device does part of every answer."
    - Change "It also turns the room's work into words" to "It also picks each next word (the model's last step)."
24. **Room card (`room/card.js`).**
    - Draw the dots mark plus lowercase "pooled" in Geist 500 with -.035em tracking, like `.brand`, instead of "Pooled".
    - Change the "ROOM ZRDJ" label to "Room" plus the code in mono.
25. **Model names.** In `room/models.js`, rename the labels to "Qwen3.8 27B · Q4" and "Qwen3.6 35B MoE · Q4".
26. **Zoom and tap targets.**
    - Remove `maximum-scale=1` from the viewport meta (p2p.html line 5).
    - Under `(pointer:coarse)`: inputs and textareas at 16px, including `#ed-text`.
    - Under `(pointer:coarse)`, give these at least 44px of hit area with `::before{inset:-8px}`: `.pchip`, `#band-toggle`, `.copy-ans`, `#menu-close`, `#name-shuffle`, `#share-close`, `#code-auto` (and its label), `#code-newtask` and `#pv-reload`.
    - On phones, widen the band's name column (`min-width:40%`) so names like "Swift Wombat" aren't cut off.
27. **Social preview tags on `p2p.html`.** Add the same `og:*` and `twitter:*` tags as the landing, pointing at `/site/og.png`.

## Deferred (new features or restructures, outside this pass)

- **Real numbers strip and "how it works / limitations" section** on the landing (developer #3, #4).
- **Saving the host and putting the code in the URL before a model starts** (first-timer #5). This is a behavior fix and belongs in its own PR.
- **"X joined" toast in the room and a pulse on the pool number** (designer #3).
- **Starter prompt chips** in the Chat and Code empty states (designer #6).
- **Log | Preview switch, or an Agent / Preview / Files control, in mobile Code mode.** Also moving the project controls into a sheet.
- **Joining automatically on the 4th character.**
- **Hiding the join-screen stepper completely.** The pass keeps it, restyled.
- **Putting the Pirate and Haiku answer styles behind a "Fun" disclosure.** Studies disagree, so they stay as is.
- **Finishing the repo rename and showing the star count** (due 2026-10-11).
- **Opening the demo on the Chat story instead of the Tetris hook.** L3 and L2 fix what made the hook look broken.

## Outcome

Landed on `feat/engine-opt` as `ui/polish-landing` (c12017f, L1 to L21) and `ui/polish-room` (8ad5fcd, R1 to R27), then `p2p.html` links `/site/css/tokens.css`, which closes the R1 dependency.

Where the work departed from this list:
- **L7:** the estimate opens at "about 4 min left", which matches about 50 MB/s, then counts down to 1 min.
- **L8:** the demo uses the room's new card titles ("New room", "Have a code?").
- **L14:** only the chat frame needed centring.
- **L16:** the hero headline keeps `clamp(38px,4.2vw,60px)` so the demo stays in view at 1280x720.
- **R3:** the popover's "Lend this device" stays a full-width secondary button.
- **R15:** the `#code-log` top fade applies only at 820px and wider, where the log scrolls.
- **Lend screen:** a device whose colour is a dark neutral shows its layers light on the dark background.
- **Not changed:** "Click to run" in `harness/preview-frame.js`.
