export function AmbientNetwork() {
  const nodes = [
    [8, 18], [22, 32], [40, 14], [61, 27], [80, 12], [93, 38],
    [13, 68], [31, 82], [49, 60], [72, 77], [88, 63],
  ];
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_10%,color-mix(in_oklab,var(--teal)_8%,transparent),transparent_32%),radial-gradient(circle_at_85%_25%,color-mix(in_oklab,var(--violet)_10%,transparent),transparent_28%)]" />
      <svg className="absolute inset-0 size-full opacity-35" viewBox="0 0 100 100" preserveAspectRatio="none">
        <g stroke="currentColor" className="text-primary/20" strokeWidth=".08">
          <path d="M8 18L22 32L40 14L61 27L80 12L93 38L88 63L72 77L49 60L31 82L13 68L22 32M61 27L49 60M22 32L49 60M93 38L72 77" />
        </g>
        {nodes.map(([x,y], i) => <circle key={i} cx={x} cy={y} r=".28" className={i % 3 === 0 ? "fill-violet" : "fill-teal"} style={{ animation: `pulse-node ${4 + i % 4}s ease-in-out ${i * .3}s infinite` }} />)}
      </svg>
      <div className="absolute inset-0 bg-[linear-gradient(to_bottom,transparent_70%,var(--background))]" />
    </div>
  );
}