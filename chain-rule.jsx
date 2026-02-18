import { useState, useEffect } from "react";

const FONT = `'JetBrains Mono', 'Fira Code', 'SF Mono', monospace`;
const DISPLAY_FONT = `'Instrument Serif', Georgia, serif`;

const C = {
  bg: "#0a0e1a",
  card: "#111827",
  border: "#1e293b",
  text: "#e2e8f0",
  muted: "#94a3b8",
  dim: "#475569",
  cyan: "#22d3ee",
  purple: "#a78bfa",
  pink: "#f472b6",
  green: "#34d399",
  orange: "#fb923c",
  red: "#ef4444",
  indigo: "#6366f1",
  yellow: "#fbbf24",
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sigmoidDeriv = (a) => a * (1 - a);

// Fixed network values for clarity
const NETWORK = {
  // Input neuron i
  a_i: 0.8,
  // Weight from i to j
  w_ij: 0.5,
  // Bias for neuron j
  b_j: 0.1,
  // Other inputs to neuron j (for realism)
  otherInputs: 0.3, // sum of other w*a contributions
  // Target
  target: 0.9,
};

// Derived values
const z_j = NETWORK.w_ij * NETWORK.a_i + NETWORK.otherInputs + NETWORK.b_j;
const a_j = sigmoid(z_j);
const loss = 0.5 * (a_j - NETWORK.target) ** 2;

// Partial derivatives
const dL_da_j = a_j - NETWORK.target;
const da_j_dz_j = sigmoidDeriv(a_j);
const dz_j_dw_ij = NETWORK.a_i;
const dL_dw_ij = dL_da_j * da_j_dz_j * dz_j_dw_ij;

const LEARNING_RATE = 0.5;
const w_new = NETWORK.w_ij - LEARNING_RATE * dL_dw_ij;

const STEPS = [
  {
    key: "overview",
    title: "The Setup",
    highlight: "all",
  },
  {
    key: "forward_z",
    title: "Step 1: Compute Weighted Sum (zⱼ)",
    highlight: "z",
  },
  {
    key: "forward_a",
    title: "Step 2: Apply Activation (aⱼ)",
    highlight: "a",
  },
  {
    key: "loss",
    title: "Step 3: Compute Loss",
    highlight: "loss",
  },
  {
    key: "dL_da",
    title: "Step 4: ∂L/∂aⱼ — How does loss change with activation?",
    highlight: "dL_da",
  },
  {
    key: "da_dz",
    title: "Step 5: ∂aⱼ/∂zⱼ — How does activation change with z?",
    highlight: "da_dz",
  },
  {
    key: "dz_dw",
    title: "Step 6: ∂zⱼ/∂wᵢⱼ — How does z change with this weight?",
    highlight: "dz_dw",
  },
  {
    key: "chain",
    title: "Step 7: Multiply — The Chain Rule",
    highlight: "chain",
  },
  {
    key: "update",
    title: "Step 8: Update the Weight",
    highlight: "update",
  },
];

function NumBox({ label, value, color, small, glow }) {
  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 2,
      }}
    >
      <span style={{ fontSize: small ? "0.65rem" : "0.7rem", color: C.dim }}>{label}</span>
      <span
        style={{
          background: `${color}15`,
          border: `1.5px solid ${color}`,
          borderRadius: 6,
          padding: "3px 10px",
          fontSize: small ? "0.85rem" : "1rem",
          fontWeight: 600,
          color,
          fontFamily: FONT,
          boxShadow: glow ? `0 0 12px ${color}40` : "none",
          transition: "all 0.4s ease",
        }}
      >
        {typeof value === "number" ? value.toFixed(4) : value}
      </span>
    </div>
  );
}

function DiagramNode({ x, y, label, value, color, r = 28, active, sublabel }) {
  return (
    <g>
      {active && (
        <circle cx={x} cy={y} r={r + 8} fill="none" stroke={color} strokeWidth={1.5} opacity={0.25}>
          <animate attributeName="r" values={`${r + 5};${r + 11};${r + 5}`} dur="1.8s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.25;0.1;0.25" dur="1.8s" repeatCount="indefinite" />
        </circle>
      )}
      <circle
        cx={x}
        cy={y}
        r={r}
        fill={C.bg}
        stroke={color}
        strokeWidth={active ? 2.5 : 1.5}
        opacity={active ? 1 : 0.5}
        style={{ transition: "all 0.4s" }}
      />
      <text x={x} y={y - (sublabel ? 5 : 0)} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize={12} fontFamily={FONT} fontWeight={500}>
        {label}
      </text>
      {sublabel && (
        <text x={x} y={y + 10} textAnchor="middle" dominantBaseline="middle" fill={C.muted} fontSize={9} fontFamily={FONT}>
          {sublabel}
        </text>
      )}
    </g>
  );
}

export default function ChainRuleWalkthrough() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const current = STEPS[step];

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => {
      setStep((s) => {
        if (s >= STEPS.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 3500);
    return () => clearTimeout(timer);
  }, [playing, step]);

  const hl = current.highlight;

  const isForward = ["overview", "forward_z", "forward_a", "loss"].includes(current.key);
  const isBackward = ["dL_da", "da_dz", "dz_dw", "chain", "update"].includes(current.key);

  const renderExplanation = () => {
    switch (current.key) {
      case "overview":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 12px" }}>We'll trace one weight through the entire forward and backward pass with real numbers.</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "center" }}>
              <NumBox label="Input aᵢ" value={NETWORK.a_i} color={C.cyan} />
              <NumBox label="Weight wᵢⱼ" value={NETWORK.w_ij} color={C.indigo} />
              <NumBox label="Bias bⱼ" value={NETWORK.b_j} color={C.dim} />
              <NumBox label="Target" value={NETWORK.target} color={C.pink} />
            </div>
            <p style={{ margin: "12px 0 0", fontSize: "0.78rem", color: C.dim, textAlign: "center" }}>
              Other inputs to neuron j contribute {NETWORK.otherInputs} to the weighted sum.
            </p>
          </div>
        );
      case "forward_z":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>Neuron j computes its weighted sum from all incoming connections:</p>
            <div style={{ background: `${C.purple}10`, border: `1px solid ${C.purple}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                z<sub>j</sub> = w<sub>ij</sub> · a<sub>i</sub> + (other inputs) + b<sub>j</sub>
              </div>
              <div style={{ fontSize: "1rem", color: C.purple }}>
                z<sub>j</sub> = {NETWORK.w_ij} × {NETWORK.a_i} + {NETWORK.otherInputs} + {NETWORK.b_j} = <strong>{z_j.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              This is just a linear combination — without an activation function, this is all the neuron could do.
            </p>
          </div>
        );
      case "forward_a":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>Now we squash z<sub>j</sub> through the sigmoid activation:</p>
            <div style={{ background: `${C.green}10`, border: `1px solid ${C.green}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                a<sub>j</sub> = σ(z<sub>j</sub>) = 1 / (1 + e<sup>−z</sup>)
              </div>
              <div style={{ fontSize: "1rem", color: C.green }}>
                a<sub>j</sub> = σ({z_j.toFixed(4)}) = <strong>{a_j.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              Sigmoid maps any value to (0, 1). This is the nonlinearity that gives hidden layers their power.
            </p>
          </div>
        );
      case "loss":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>How far off is our prediction? Using MSE loss:</p>
            <div style={{ background: `${C.red}10`, border: `1px solid ${C.red}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                L = ½(a<sub>j</sub> − target)²
              </div>
              <div style={{ fontSize: "1rem", color: C.red }}>
                L = ½({a_j.toFixed(4)} − {NETWORK.target})² = <strong>{loss.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              Now we need to figure out: how should we change w<sub>ij</sub> to make this loss smaller?
            </p>
          </div>
        );
      case "dL_da":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>Starting from the loss, how sensitive is it to neuron j's output?</p>
            <div style={{ background: `${C.orange}10`, border: `1px solid ${C.orange}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                ∂L/∂a<sub>j</sub> = a<sub>j</sub> − target
              </div>
              <div style={{ fontSize: "1rem", color: C.orange }}>
                ∂L/∂a<sub>j</sub> = {a_j.toFixed(4)} − {NETWORK.target} = <strong>{dL_da_j.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              Negative value → our prediction was too low → we need to increase a<sub>j</sub> to reduce loss.
            </p>
          </div>
        );
      case "da_dz":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>How sensitive is the activation to changes in the pre-activation z?</p>
            <div style={{ background: `${C.yellow}10`, border: `1px solid ${C.yellow}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                ∂a<sub>j</sub>/∂z<sub>j</sub> = σ(z) · (1 − σ(z)) = a<sub>j</sub> · (1 − a<sub>j</sub>)
              </div>
              <div style={{ fontSize: "1rem", color: C.yellow }}>
                ∂a<sub>j</sub>/∂z<sub>j</sub> = {a_j.toFixed(4)} × (1 − {a_j.toFixed(4)}) = <strong>{da_j_dz_j.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              This is the sigmoid's derivative. Notice: at extreme values (near 0 or 1), this approaches 0 — the vanishing gradient problem.
            </p>
          </div>
        );
      case "dz_dw":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>How sensitive is the weighted sum to this specific weight?</p>
            <div style={{ background: `${C.cyan}10`, border: `1px solid ${C.cyan}30`, borderRadius: 8, padding: "12px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 6 }}>
                z<sub>j</sub> = w<sub>ij</sub> · a<sub>i</sub> + ... &nbsp; → &nbsp; ∂z<sub>j</sub>/∂w<sub>ij</sub> = a<sub>i</sub>
              </div>
              <div style={{ fontSize: "1rem", color: C.cyan }}>
                ∂z<sub>j</sub>/∂w<sub>ij</sub> = a<sub>i</sub> = <strong>{dz_j_dw_ij.toFixed(4)}</strong>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              It's just the input activation! The gradient of a weight is always the upstream neuron's value times the downstream error.
            </p>
          </div>
        );
      case "chain":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>Now multiply all three terms together:</p>
            <div style={{ background: `${C.indigo}10`, border: `1px solid ${C.indigo}30`, borderRadius: 8, padding: "14px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 8 }}>
                ∂L/∂w<sub>ij</sub> = <span style={{ color: C.orange }}>∂L/∂a<sub>j</sub></span> · <span style={{ color: C.yellow }}>∂a<sub>j</sub>/∂z<sub>j</sub></span> · <span style={{ color: C.cyan }}>∂z<sub>j</sub>/∂w<sub>ij</sub></span>
              </div>
              <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 8, flexWrap: "wrap", fontSize: "1rem" }}>
                <span style={{ color: C.orange }}>{dL_da_j.toFixed(4)}</span>
                <span style={{ color: C.dim }}>×</span>
                <span style={{ color: C.yellow }}>{da_j_dz_j.toFixed(4)}</span>
                <span style={{ color: C.dim }}>×</span>
                <span style={{ color: C.cyan }}>{dz_j_dw_ij.toFixed(4)}</span>
                <span style={{ color: C.dim }}>=</span>
                <span style={{ color: C.indigo, fontWeight: 700, fontSize: "1.1rem" }}>{dL_dw_ij.toFixed(4)}</span>
              </div>
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              This gradient tells us: nudging w<sub>ij</sub> up slightly will change the loss by ≈ {dL_dw_ij.toFixed(4)}.
            </p>
          </div>
        );
      case "update":
        return (
          <div style={{ lineHeight: 1.9 }}>
            <p style={{ margin: "0 0 10px" }}>Finally, step in the direction that reduces loss:</p>
            <div style={{ background: `${C.green}10`, border: `1px solid ${C.green}30`, borderRadius: 8, padding: "14px 16px", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", color: C.muted, marginBottom: 8 }}>
                w<sub>ij</sub><sup>new</sup> = w<sub>ij</sub> − η · ∂L/∂w<sub>ij</sub>
              </div>
              <div style={{ fontSize: "1rem", color: C.green }}>
                w<sub>ij</sub><sup>new</sup> = {NETWORK.w_ij} − {LEARNING_RATE} × ({dL_dw_ij.toFixed(4)}) = <strong>{w_new.toFixed(4)}</strong>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 12 }}>
              <NumBox label="Old weight" value={NETWORK.w_ij} color={C.dim} small />
              <span style={{ color: C.green, fontSize: "1.4rem", alignSelf: "flex-end", paddingBottom: 2 }}>→</span>
              <NumBox label="New weight" value={w_new} color={C.green} small glow />
            </div>
            <p style={{ margin: "10px 0 0", fontSize: "0.78rem", color: C.dim }}>
              The weight increased (since gradient was negative) — pushing the prediction closer to the target. Learning rate η = {LEARNING_RATE} controls the step size.
            </p>
          </div>
        );
      default:
        return null;
    }
  };

  // Mini computation graph for the SVG
  const nodes = [
    { id: "ai", x: 60, y: 100, label: "aᵢ", sub: NETWORK.a_i.toFixed(1), color: C.cyan },
    { id: "wij", x: 165, y: 50, label: "wᵢⱼ", sub: NETWORK.w_ij.toFixed(1), color: C.indigo },
    { id: "zj", x: 270, y: 100, label: "zⱼ", sub: z_j.toFixed(2), color: C.purple },
    { id: "aj", x: 400, y: 100, label: "aⱼ", sub: a_j.toFixed(3), color: C.green },
    { id: "L", x: 530, y: 100, label: "L", sub: loss.toFixed(4), color: C.red },
  ];

  const edges = [
    { from: "ai", to: "zj", label: "× wᵢⱼ" },
    { from: "wij", to: "zj", label: "" },
    { from: "zj", to: "aj", label: "σ(·)" },
    { from: "aj", to: "L", label: "MSE" },
  ];

  const nodeMap = {};
  nodes.forEach((n) => (nodeMap[n.id] = n));

  const getNodeActive = (id) => {
    if (hl === "all") return true;
    if (hl === "z" && (id === "ai" || id === "wij" || id === "zj")) return true;
    if (hl === "a" && (id === "zj" || id === "aj")) return true;
    if (hl === "loss" && (id === "aj" || id === "L")) return true;
    if (hl === "dL_da" && (id === "aj" || id === "L")) return true;
    if (hl === "da_dz" && (id === "zj" || id === "aj")) return true;
    if (hl === "dz_dw" && (id === "ai" || id === "wij" || id === "zj")) return true;
    if (hl === "chain" || hl === "update") return true;
    return false;
  };

  const getEdgeActive = (from, to) => {
    if (hl === "all" || hl === "chain" || hl === "update") return "neutral";
    if (hl === "z" && to === "zj") return "forward";
    if (hl === "a" && from === "zj" && to === "aj") return "forward";
    if (hl === "loss" && from === "aj" && to === "L") return "forward";
    if (hl === "dL_da" && from === "aj" && to === "L") return "backward";
    if (hl === "da_dz" && from === "zj" && to === "aj") return "backward";
    if (hl === "dz_dw" && to === "zj") return "backward";
    return "dim";
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: C.text,
        fontFamily: FONT,
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 18,
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=JetBrains+Mono:wght@300;400;500;600&display=swap"
        rel="stylesheet"
      />

      {/* Title */}
      <div style={{ textAlign: "center" }}>
        <h1
          style={{
            fontFamily: DISPLAY_FONT,
            fontSize: "2rem",
            fontWeight: 400,
            margin: 0,
            background: `linear-gradient(135deg, ${C.orange}, ${C.yellow}, ${C.green})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Chain Rule Walkthrough
        </h1>
        <p style={{ color: C.muted, fontSize: "0.8rem", margin: "4px 0 0" }}>
          Tracing one weight through forward pass → loss → backprop → update
        </p>
      </div>

      {/* Computation graph */}
      <div
        style={{
          background: C.card,
          border: `1px solid ${C.border}`,
          borderRadius: 14,
          width: "100%",
          maxWidth: 620,
          overflow: "hidden",
        }}
      >
        <svg viewBox="0 0 590 170" style={{ width: "100%", display: "block" }}>
          {/* Edges */}
          {edges.map((e, i) => {
            const from = nodeMap[e.from];
            const to = nodeMap[e.to];
            const state = getEdgeActive(e.from, e.to);
            const color =
              state === "forward" ? C.green : state === "backward" ? C.orange : state === "neutral" ? C.dim : `${C.dim}40`;
            const width = state === "dim" ? 1 : 2;
            const mx = (from.x + to.x) / 2;
            const my = e.from === "wij" ? 70 : (from.y + to.y) / 2 - 14;

            return (
              <g key={i}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={color}
                  strokeWidth={width}
                  opacity={state === "dim" ? 0.3 : 0.8}
                  style={{ transition: "all 0.4s" }}
                />
                {e.label && state !== "dim" && (
                  <text x={mx} y={my} textAnchor="middle" fill={color} fontSize={9} fontFamily={FONT} opacity={0.8}>
                    {e.label}
                  </text>
                )}
                {state === "backward" && (
                  <g>
                    <circle r={3} fill={C.orange} opacity={0.9}>
                      <animateMotion
                        dur="1s"
                        repeatCount="indefinite"
                        path={`M${to.x},${to.y} L${from.x},${from.y}`}
                      />
                    </circle>
                  </g>
                )}
                {state === "forward" && (
                  <g>
                    <circle r={3} fill={C.green} opacity={0.9}>
                      <animateMotion
                        dur="1s"
                        repeatCount="indefinite"
                        path={`M${from.x},${from.y} L${to.x},${to.y}`}
                      />
                    </circle>
                  </g>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {nodes.map((n) => (
            <DiagramNode
              key={n.id}
              x={n.x}
              y={n.y}
              label={n.label}
              sublabel={n.sub}
              color={n.color}
              r={24}
              active={getNodeActive(n.id)}
            />
          ))}

          {/* Direction label */}
          {isForward && step > 0 && (
            <text x={295} y={160} textAnchor="middle" fill={C.green} fontSize={10} fontFamily={FONT} opacity={0.6}>
              → Forward →
            </text>
          )}
          {isBackward && (
            <text x={295} y={160} textAnchor="middle" fill={C.orange} fontSize={10} fontFamily={FONT} opacity={0.6}>
              ← Backward ←
            </text>
          )}
        </svg>
      </div>

      {/* Step info */}
      <div
        style={{
          background: C.card,
          border: `1px solid ${C.border}`,
          borderRadius: 12,
          padding: "18px 24px",
          maxWidth: 620,
          width: "100%",
        }}
      >
        <div
          style={{
            fontSize: "0.95rem",
            fontWeight: 500,
            marginBottom: 12,
            color: isBackward ? C.orange : current.key === "update" ? C.green : C.text,
          }}
        >
          {current.title}
        </div>
        <div style={{ fontSize: "0.82rem", color: C.muted }}>{renderExplanation()}</div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <button
          onClick={() => { setStep(0); setPlaying(false); }}
          style={btnStyle("#334155")}
        >
          ↺ Reset
        </button>
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          style={btnStyle("#475569", step === 0)}
        >
          ← Back
        </button>
        <button
          onClick={() => {
            if (step >= STEPS.length - 1) {
              setStep(0);
              setPlaying(true);
            } else {
              setPlaying(!playing);
            }
          }}
          style={btnStyle(playing ? "#dc2626" : "#059669")}
        >
          {playing ? "⏸ Pause" : step >= STEPS.length - 1 ? "↻ Replay" : "▶ Play"}
        </button>
        <button
          onClick={() => { setPlaying(false); setStep((s) => Math.min(s + 1, STEPS.length - 1)); }}
          disabled={step >= STEPS.length - 1}
          style={btnStyle("#4f46e5", step >= STEPS.length - 1)}
        >
          Next →
        </button>
      </div>

      {/* Progress */}
      <div style={{ width: "100%", maxWidth: 620, display: "flex", gap: 3 }}>
        {STEPS.map((s, i) => {
          const isBack = ["dL_da", "da_dz", "dz_dw", "chain"].includes(s.key);
          const isUpd = s.key === "update";
          return (
            <div
              key={i}
              onClick={() => { setStep(i); setPlaying(false); }}
              style={{
                flex: 1,
                height: 5,
                borderRadius: 3,
                background:
                  i <= step
                    ? isBack
                      ? C.orange
                      : isUpd
                      ? C.green
                      : i === step
                      ? C.text
                      : C.green
                    : C.border,
                cursor: "pointer",
                opacity: i <= step ? 1 : 0.3,
                transition: "all 0.3s",
              }}
            />
          );
        })}
      </div>

      {/* Summary equation */}
      <div
        style={{
          background: C.card,
          border: `1px solid ${C.border}`,
          borderRadius: 10,
          padding: "12px 20px",
          maxWidth: 620,
          width: "100%",
          textAlign: "center",
          fontSize: "0.75rem",
          color: C.dim,
        }}
      >
        <span style={{ color: C.muted }}>Full chain: </span>
        <span style={{ color: C.orange }}>({dL_da_j.toFixed(3)})</span>
        <span style={{ color: C.dim }}> × </span>
        <span style={{ color: C.yellow }}>({da_j_dz_j.toFixed(3)})</span>
        <span style={{ color: C.dim }}> × </span>
        <span style={{ color: C.cyan }}>({dz_j_dw_ij.toFixed(3)})</span>
        <span style={{ color: C.dim }}> = </span>
        <span style={{ color: C.indigo, fontWeight: 600 }}>{dL_dw_ij.toFixed(4)}</span>
        <span style={{ color: C.dim }}> → </span>
        <span style={{ color: C.green, fontWeight: 600 }}>w = {w_new.toFixed(4)}</span>
      </div>
    </div>
  );
}

const COLORS = {
  bg: "#0a0e1a",
};

const btnStyle = (bg, disabled) => ({
  background: disabled ? "#1e293b" : bg,
  color: disabled ? "#475569" : "#fff",
  border: "none",
  borderRadius: 8,
  padding: "8px 16px",
  fontSize: "0.78rem",
  fontFamily: `'JetBrains Mono', monospace`,
  fontWeight: 500,
  cursor: disabled ? "not-allowed" : "pointer",
  opacity: disabled ? 0.5 : 1,
  transition: "all 0.2s",
});
