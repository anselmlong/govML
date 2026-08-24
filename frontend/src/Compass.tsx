interface CompassProps {
  size?: number;
  spinning?: boolean;
  signature?: boolean;
  className?: string;
}

export default function Compass({ size = 16, spinning = false, signature = false, className = '' }: CompassProps) {
  const cls = ['compass-glyph', spinning ? 'is-spinning' : '', signature ? 'is-signature' : '', className].filter(Boolean).join(' ');
  return (
    <svg
      className={cls}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="16" cy="16" r="14.25" stroke="currentColor" strokeWidth="1.1" opacity={signature ? 1 : 0.85} />
      <g stroke="currentColor" strokeWidth="1" opacity="0.6">
        <path d="M16 2.4V5.6" />
        <path d="M16 26.4V29.6" />
        <path d="M2.4 16H5.6" />
        <path d="M26.4 16H29.6" />
        <path d="M6.7 6.7L8.9 8.9" />
        <path d="M23.1 23.1L25.3 25.3" />
        <path d="M6.7 25.3L8.9 23.1" />
        <path d="M23.1 8.9L25.3 6.7" />
      </g>
      <g className="compass-needle">
        <path d="M16 6.5L19.6 16L16 15.1Z" className="compass-needle-fwd" />
        <path d="M16 25.5L12.4 16L16 16.9Z" className="compass-needle-aft" />
      </g>
      <circle cx="16" cy="16" r="1.6" fill="currentColor" />
    </svg>
  );
}
