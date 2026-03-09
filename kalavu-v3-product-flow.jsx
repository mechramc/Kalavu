import { useState } from "react";

const phases = [
  { id: "create", label: "Create Cooperative", cmd: "kalavu coop create", actor: "Organizer", color: "#A78BFA" },
  { id: "train", label: "Train Module", cmd: "kalavu train start", actor: "Contributor", color: "#E8B931" },
  { id: "check", label: "Alignment Check", cmd: "kalavu check post", actor: "Automatic", color: "#4ECDC4" },
  { id: "fuse", label: "Fuse Model", cmd: "kalavu fuse build", actor: "Organizer", color: "#FF6B6B" },
];

const CreateDiagram = () => (
  <svg viewBox="0 0 700 320" style={{ width: "100%", height: "auto" }}>
    <text x={350} y={25} textAnchor="middle" fontSize={11} fill="#A78BFA" fontWeight={700} fontFamily="'JetBrains Mono', monospace">
      kalavu coop create --name "open-20b" --modules 20
    </text>
    <rect x={200} y={40} width={300} height={40} rx={8} fill="#A78BFA" fillOpacity={0.1} stroke="#A78BFA" strokeWidth={2} />
    <text x={350} y={65} textAnchor="middle" fontSize={10} fill="#A78BFA" fontWeight={600} fontFamily="'JetBrains Mono', monospace">COOPERATIVE GENESIS</text>

    {[
      { x: 40, y: 110, w: 120, label: "minbpe", sub: "Train shared\ntokenizer", icon: "T" },
      { x: 180, y: 110, w: 120, label: "nanochat init", sub: "Generate\nseed \u03B8\u2080", icon: "\u03B8" },
      { x: 320, y: 110, w: 120, label: "CKA Reference", sub: "Compute\nalignment target", icon: "R" },
      { x: 460, y: 110, w: 120, label: "Domain Manifest", sub: "20 slots with\ndata hints", icon: "D" },
    ].map((item, idx) => (
      <g key={idx}>
        <line x1={350} y1={80} x2={item.x + item.w/2} y2={110} stroke="#A78BFA" strokeWidth={1} strokeOpacity={0.4} />
        <rect x={item.x} y={item.y} width={item.w} height={80} rx={8} fill="#A78BFA" fillOpacity={0.06} stroke="#A78BFA" strokeWidth={1.2} />
        <circle cx={item.x + 20} cy={item.y + 20} r={12} fill="#A78BFA" fillOpacity={0.15} />
        <text x={item.x + 20} y={item.y + 24} textAnchor="middle" fontSize={10} fill="#A78BFA" fontWeight={700} fontFamily="'JetBrains Mono', monospace">{item.icon}</text>
        <text x={item.x + item.w/2} y={item.y + 40} textAnchor="middle" fontSize={9} fill="#A78BFA" fontWeight={600} fontFamily="'JetBrains Mono', monospace">{item.label}</text>
        {item.sub.split("\n").map((l, li) => (
          <text key={li} x={item.x + item.w/2} y={item.y + 55 + li * 12} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">{l}</text>
        ))}
      </g>
    ))}

    {/* Published to GitHub */}
    <rect x={100} y={220} width={500} height={80} rx={12} fill="#1a1a2e" stroke="#A78BFA" strokeWidth={1} strokeDasharray="4,3" />
    <text x={350} y={245} textAnchor="middle" fontSize={10} fill="#A78BFA" fontWeight={700} fontFamily="'JetBrains Mono', monospace">COOPERATIVE REPOSITORY (GitHub)</text>
    <text x={350} y={265} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">kalavu.yaml \u00B7 tokenizer.model \u00B7 seed_checkpoint.pt \u00B7 calibration_batch.pt \u00B7 cka_reference.pt</text>
    <text x={350} y={282} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">domain_manifest.json \u00B7 20 open module slots \u00B7 README with contributor instructions</text>
    {[0,1,2,3].map(idx => (
      <line key={idx} x1={100 + idx * 140 + 60} y1={190} x2={100 + idx * 140 + 60} y2={220} stroke="#A78BFA" strokeWidth={1} strokeOpacity={0.3} />
    ))}
  </svg>
);

const TrainDiagram = () => (
  <svg viewBox="0 0 700 360" style={{ width: "100%", height: "auto" }}>
    <text x={350} y={20} textAnchor="middle" fontSize={10} fill="#E8B931" fontWeight={700} fontFamily="'JetBrains Mono', monospace">
      20 CONTRIBUTORS \u00D7 20 GPUs \u00D7 FULLY ASYNCHRONOUS
    </text>

    {/* Contributors */}
    {Array.from({length: 5}).map((_, row) => (
      Array.from({length: 4}).map((_, col) => {
        const idx = row * 4 + col;
        const x = 40 + col * 165;
        const y = 40 + row * 60;
        const names = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Iris","Jack",
                       "Kate","Leo","Mia","Nick","Olga","Pete","Quinn","Rose","Sam","Tina"];
        const hw = ["5090","H100","4090","M4 Max","A100","5090","3090","H100","4090","5090",
                     "M4","H100","5090","A100","4090","5090","H100","M4 Max","A100","5090"];
        const domains = ["Code","Math","Bio","Legal","History","Physics","NLP","Logic","Medical",
                        "Finance","Chem","Writing","Dialogue","Causal","Spatial","Ethics","News","Geo","Multi-L","Multi-H"];
        const pcts = [95,72,88,41,63,55,100,30,82,47,68,15,77,60,90,35,50,85,25,70];
        const done = pcts[idx] >= 100;
        return (
          <g key={idx}>
            <rect x={x} y={y} width={150} height={48} rx={6}
              fill={done ? "#4ECDC4" : "#E8B931"} fillOpacity={done ? 0.08 : 0.04}
              stroke={done ? "#4ECDC4" : "#E8B931"} strokeWidth={1} />
            <text x={x + 8} y={y + 15} fontSize={8} fill={done ? "#4ECDC4" : "#E8B931"} fontWeight={700} fontFamily="'JetBrains Mono', monospace">
              M{idx + 1}: {domains[idx]}
            </text>
            <text x={x + 8} y={y + 28} fontSize={7} fill="#888" fontFamily="'JetBrains Mono', monospace">
              {names[idx]} \u00B7 {hw[idx]}
            </text>
            <rect x={x + 8} y={y + 35} width={Math.min(pcts[idx], 100) * 1.3} height={5} rx={2}
              fill={done ? "#4ECDC4" : "#E8B931"} fillOpacity={0.5} />
            <text x={x + 142} y={y + 41} textAnchor="end" fontSize={7} fill="#666" fontFamily="'JetBrains Mono', monospace">
              {done ? "\u2713" : pcts[idx] + "%"}
            </text>
          </g>
        );
      })
    ))}

    <text x={350} y={350} textAnchor="middle" fontSize={9} fill="#666" fontStyle="italic" fontFamily="'JetBrains Mono', monospace">
      Each contributor trains independently. No synchronization required. Submit when done.
    </text>
  </svg>
);

const CheckDiagram = () => (
  <svg viewBox="0 0 700 300" style={{ width: "100%", height: "auto" }}>
    <text x={350} y={25} textAnchor="middle" fontSize={11} fill="#4ECDC4" fontWeight={700} fontFamily="'JetBrains Mono', monospace">
      ALIGNMENT MONITORING (GitHub Discussions)
    </text>

    {/* Module telemetry posts */}
    {[0,1,2,3,4].map(idx => {
      const x = 20 + idx * 136;
      const cka = [0.82, 0.75, 0.91, 0.43, 0.88];
      const ok = cka[idx] > 0.6;
      return (
        <g key={idx}>
          <rect x={x} y={45} width={125} height={130} rx={8}
            fill={ok ? "#4ECDC4" : "#FF6B6B"} fillOpacity={0.05}
            stroke={ok ? "#4ECDC4" : "#FF6B6B"} strokeWidth={1.5} />
          <text x={x + 62} y={65} textAnchor="middle" fontSize={9} fill={ok ? "#4ECDC4" : "#FF6B6B"} fontWeight={700} fontFamily="'JetBrains Mono', monospace">
            Module {idx + 1}
          </text>
          <text x={x + 62} y={85} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">CKA@L6: {(cka[idx] + 0.05).toFixed(2)}</text>
          <text x={x + 62} y={100} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">CKA@L12: {cka[idx].toFixed(2)}</text>
          <text x={x + 62} y={115} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">CKA@L18: {(cka[idx] - 0.1).toFixed(2)}</text>
          <rect x={x + 15} y={125} width={95} height={20} rx={4}
            fill={ok ? "#4ECDC4" : "#FF6B6B"} fillOpacity={0.12} />
          <text x={x + 62} y={139} textAnchor="middle" fontSize={8} fill={ok ? "#4ECDC4" : "#FF6B6B"} fontWeight={700} fontFamily="'JetBrains Mono', monospace">
            {ok ? "\u2713 ALIGNED" : "\u26A0 DIVERGING"}
          </text>
          {!ok && (
            <text x={x + 62} y={160} textAnchor="middle" fontSize={7} fill="#FF6B6B" fontFamily="'JetBrains Mono', monospace">
              Action: increase \u03BB, lower LR
            </text>
          )}
        </g>
      );
    })}

    <rect x={50} y={195} width={600} height={80} rx={10} fill="#1a1a2e" stroke="#4ECDC4" strokeWidth={1} strokeDasharray="4,3" />
    <text x={350} y={220} textAnchor="middle" fontSize={10} fill="#4ECDC4" fontWeight={700} fontFamily="'JetBrains Mono', monospace">kalavu coop status open-20b</text>
    <text x={350} y={240} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">18/20 modules aligned \u00B7 14/20 training \u00B7 4/20 submitted \u00B7 2/20 flagged</text>
    <text x={350} y={258} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">Est. cooperative completion: 6 days \u00B7 Fusion eligible: when 10+ modules submitted</text>
  </svg>
);

const FuseDiagram = () => (
  <svg viewBox="0 0 700 330" style={{ width: "100%", height: "auto" }}>
    <text x={350} y={25} textAnchor="middle" fontSize={11} fill="#FF6B6B" fontWeight={700} fontFamily="'JetBrains Mono', monospace">
      FUSION PIPELINE
    </text>

    {/* Three steps */}
    {[
      { x: 40, label: "kalavu fuse cluster", sub: "Pairwise CKA \u2192\nautomated clustering\n\u2192 4 groups", color: "#FF6B6B" },
      { x: 265, label: "kalavu fuse build", sub: "Assemble fusion\narchitecture\n(MoE or XAttn)", color: "#E8B931" },
      { x: 490, label: "kalavu fuse train", sub: "Post-training\ncurriculum\n8-12% of pretrain cost", color: "#4ECDC4" },
    ].map((s, idx) => (
      <g key={idx}>
        <rect x={s.x} y={45} width={195} height={100} rx={10} fill={s.color} fillOpacity={0.06} stroke={s.color} strokeWidth={1.5} />
        <text x={s.x + 97} y={70} textAnchor="middle" fontSize={9} fill={s.color} fontWeight={700} fontFamily="'JetBrains Mono', monospace">{s.label}</text>
        {s.sub.split("\n").map((l, li) => (
          <text key={li} x={s.x + 97} y={90 + li * 14} textAnchor="middle" fontSize={8} fill="#888" fontFamily="'JetBrains Mono', monospace">{l}</text>
        ))}
        {idx < 2 && <text x={s.x + 210} y={95} fontSize={16} fill="#666">{"\u2192"}</text>}
      </g>
    ))}

    {/* Output */}
    <rect x={150} y={175} width={400} height={70} rx={12} fill="#1a1a2e" stroke="#4ECDC4" strokeWidth={2} />
    <text x={350} y={200} textAnchor="middle" fontSize={12} fill="#4ECDC4" fontWeight={700} fontFamily="'JetBrains Mono', monospace">FUSED MODEL: open-20b v1</text>
    <text x={350} y={218} textAnchor="middle" fontSize={9} fill="#888" fontFamily="'JetBrains Mono', monospace">~20B+ params \u00B7 HF-compatible \u00B7 Serve with vLLM / llama.cpp</text>
    <text x={350} y={232} textAnchor="middle" fontSize={8} fill="#A78BFA" fontFamily="'JetBrains Mono', monospace">Auto-generated model card: all contributors, domains, scores</text>

    {/* Distribution */}
    <text x={350} y={270} textAnchor="middle" fontSize={10} fill="#FF6B6B" fontWeight={600} fontFamily="'JetBrains Mono', monospace">kalavu coop publish open-20b</text>
    <text x={350} y={290} textAnchor="middle" fontSize={9} fill="#888" fontFamily="'JetBrains Mono', monospace">\u2192 Hugging Face Hub \u00B7 All cooperative members get access</text>
    <text x={350} y={310} textAnchor="middle" fontSize={8} fill="#666" fontStyle="italic" fontFamily="'JetBrains Mono', monospace">New modules can be added later \u2192 incremental re-fusion \u2192 the cooperative grows</text>
  </svg>
);

const diagrams = { create: CreateDiagram, train: TrainDiagram, check: CheckDiagram, fuse: FuseDiagram };

export default function KALAVUv3() {
  const [active, setActive] = useState("create");
  const Diagram = diagrams[active];
  const phase = phases.find(p => p.id === active);

  return (
    <div style={{
      minHeight: "100vh", background: "#0d0d1a", color: "#e0e0e0",
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace", padding: "32px 24px",
    }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');`}</style>

      <div style={{ textAlign: "center", marginBottom: 36 }}>
        <div style={{ fontSize: 10, letterSpacing: 4, color: "#666", textTransform: "uppercase" }}>Murai Labs</div>
        <h1 style={{ fontSize: 32, fontWeight: 700, color: "#fff", margin: "4px 0" }}>KALAVU</h1>
        <div style={{ fontSize: 13, color: "#E8B931", marginBottom: 6 }}>The Decentralized LLM Training Protocol</div>
        <div style={{ fontSize: 11, color: "#888" }}>20 people. 20 GPUs. 1 model none of them could build alone.</div>
      </div>

      {/* Phase tabs */}
      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 28, flexWrap: "wrap" }}>
        {phases.map(ph => (
          <button key={ph.id} onClick={() => setActive(ph.id)} style={{
            background: active === ph.id ? ph.color + "18" : "transparent",
            border: `1.5px solid ${active === ph.id ? ph.color : "#333"}`,
            borderRadius: 8, padding: "8px 16px",
            color: active === ph.id ? ph.color : "#666",
            fontSize: 10, fontWeight: 600, fontFamily: "inherit", cursor: "pointer",
          }}>
            <div>{ph.label}</div>
            <div style={{ fontSize: 8, opacity: 0.6, marginTop: 2 }}>{ph.cmd}</div>
          </button>
        ))}
      </div>

      {/* Diagram */}
      <div style={{
        background: "#111125", border: `1px solid ${phase.color}33`,
        borderRadius: 12, padding: "20px 16px", maxWidth: 760, margin: "0 auto 28px",
      }}>
        <Diagram />
      </div>

      {/* Stats bar */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
        gap: 8, maxWidth: 760, margin: "0 auto",
      }}>
        {[
          { label: "Contributor Cost", value: "~$100-300", sub: "GPU electricity", color: "#E8B931" },
          { label: "Equivalent Model", value: "$5-10K", sub: "if trained alone", color: "#FF6B6B" },
          { label: "Cost Reduction", value: "20-50\u00D7", sub: "per contributor", color: "#4ECDC4" },
          { label: "Commands", value: "2", sub: "to contribute", color: "#A78BFA" },
          { label: "Coordination", value: "0", sub: "synchronization needed", color: "#E8B931" },
        ].map((m, i) => (
          <div key={i} style={{
            background: "#111125", border: "1px solid #222", borderRadius: 8,
            padding: "10px 12px", textAlign: "center",
          }}>
            <div style={{ fontSize: 7, color: "#666", letterSpacing: 1, textTransform: "uppercase" }}>{m.label}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: m.color, margin: "2px 0" }}>{m.value}</div>
            <div style={{ fontSize: 7, color: "#555" }}>{m.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
