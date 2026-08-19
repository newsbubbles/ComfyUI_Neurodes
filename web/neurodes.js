import { app } from "../../scripts/app.js";

/**
 * neurodes frontend.
 *
 * Two small jobs, both about making the graph readable at a glance:
 *
 *  1. After a run, every neurodes node shows what it produced — the output shape, and the
 *     parameter count for layers that have weights. Shape inference happens on the server
 *     in milliseconds because nothing is actually computed, so this is effectively live.
 *     "What shape is it here?" is the question this whole pack exists to answer, and it
 *     should not require clicking anything.
 *
 *  2. Each socket type gets its own colour, so a tensor path, a dataset path and a model
 *     path are told apart by looking rather than by reading.
 */

const TYPE_COLOURS = {
  NEURO_TENSOR: "#5ea8ff",   // blue    — the data flowing through the network
  NEURO_SHAPE: "#78d694",    // green   — a description of a size
  NEURO_MODEL: "#ff8a60",    // orange  — something with weights
  NEURO_DATASET: "#f0ca60",  // yellow  — examples to learn from
  NEURO_HISTORY: "#a89cff",  // violet  — what happened during training
  NEURO_TRAINER: "#6ee2e0",  // cyan    — how to train
};

const BADGE_BG = "#1b1c21";
const BADGE_BORDER = "#3a3d47";
const BADGE_TEXT = "#d0d4de";
const BADGE_ACCENT = "#5ea8ff";

/**
 * ComfyUI registers every custom socket type with an empty colour, and it does so after
 * an extension's setup() has run — so setting these once is not enough, the blanks land
 * on top. Re-applying is cheap and idempotent, so we do it whenever the graph changes.
 */
function applyTypeColours() {
  const canvas = app.canvas;
  if (!canvas) return;
  const LG = window.LGraphCanvas;
  for (const [type, colour] of Object.entries(TYPE_COLOURS)) {
    if (canvas.default_connection_color_byType) {
      canvas.default_connection_color_byType[type] = colour;
    }
    if (canvas.default_connection_color_byTypeOff) {
      canvas.default_connection_color_byTypeOff[type] = colour;
    }
    if (LG?.link_type_colors) LG.link_type_colors[type] = colour;
  }
}

function coloursMissing() {
  const byType = app.canvas?.default_connection_color_byType;
  return !byType || byType.NEURO_TENSOR !== TYPE_COLOURS.NEURO_TENSOR;
}

/** Draw a small pill just under the node with whatever the node reported. */
function drawBadge(node, ctx) {
  const text = node.__neurodesBadge;
  if (!text || node.flags?.collapsed) return;

  ctx.save();
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  const metrics = ctx.measureText(text);
  const padX = 7;
  const height = 17;
  const width = Math.min(metrics.width + padX * 2 + 8, Math.max(node.size[0], 120));
  const x = 0;
  const y = node.size[1] + 4;

  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 4);
  ctx.fillStyle = BADGE_BG;
  ctx.fill();
  ctx.strokeStyle = BADGE_BORDER;
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.beginPath();
  ctx.roundRect(x + 1.5, y + 3, 2.5, height - 6, 1.5);
  ctx.fillStyle = BADGE_ACCENT;
  ctx.fill();

  ctx.fillStyle = BADGE_TEXT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const room = width - padX - 6;
  let shown = text;
  while (shown.length > 4 && ctx.measureText(shown).width > room) {
    shown = shown.slice(0, -2);
  }
  if (shown !== text) shown = shown.slice(0, -1) + "…";
  ctx.fillText(shown, x + padX, y + height / 2 + 0.5);
  ctx.restore();
}

/** The first line of a node's text output is the badge; the rest is detail. */
function badgeFrom(message) {
  const value = message?.text;
  const raw = Array.isArray(value) ? value[0] : value;
  if (typeof raw !== "string" || !raw.trim()) return null;
  return raw.trim().split("\n")[0].slice(0, 120);
}

app.registerExtension({
  name: "neurodes.shapeBadges",

  setup() {
    applyTypeColours();
  },

  afterConfigureGraph() {
    applyTypeColours();
  },

  nodeCreated(node) {
    if (node?.type?.startsWith?.("Neuro") && coloursMissing()) applyTypeColours();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData?.name?.startsWith?.("Neuro")) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const badge = badgeFrom(message);
      if (badge) {
        this.__neurodesBadge = badge;
        this.setDirtyCanvas(true, false);
      }
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      drawBadge(this, ctx);
    };

    // A stale shape is worse than no shape: drop it as soon as the node is reconfigured.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      this.__neurodesBadge = null;
      return onConnectionsChange?.apply(this, arguments);
    };
  },
});
