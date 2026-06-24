/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          900: '#0D1117',
          800: '#161B22',
          700: '#21262D',
          600: '#30363D',
          500: '#8B949E',
          400: '#C9D1D9',
        },
        la28: {
          blue:  '#0066CC',
          navy:  '#003F8A',
          light: '#58A6FF',
        },
        ops: {
          red:   '#DC2626',
          amber: '#F59E0B',
          green: '#16A34A',
          blue:  '#58A6FF',
          white: '#F0F6FC',
          gray:  '#8B949E',
        },
        accent: {
          blue:   '#0066CC',
          teal:   '#0891b2',
          amber:  '#F59E0B',
          red:    '#DC2626',
          green:  '#16A34A',
          purple: '#7c3aed',
          slate:  '#8B949E',
        },
        olympic: {
          blue:   '#0081C8',
          yellow: '#FCB131',
          black:  '#000000',
          green:  '#00A651',
          red:    '#EE334E',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow':  'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ping-slow':   'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
        'status-blink':'status-blink 2s ease-in-out infinite',
      },
      keyframes: {
        'status-blink': {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.3' },
        },
      },
    },
  },
  plugins: [],
};
