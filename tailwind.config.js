/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          900: '#060c18',  // darkest — map/page background
          800: '#0c1628',  // card panels — slightly lifted
          700: '#182438',  // visible borders
          600: '#243448',  // secondary dividers
          500: '#3d5270',
          400: '#64748b',
        },
        accent: {
          blue:   '#2563eb',
          teal:   '#0891b2',
          amber:  '#d97706',
          red:    '#dc2626',
          green:  '#16a34a',
          purple: '#7c3aed',
          slate:  '#475569',
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
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ping-slow':  'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
    },
  },
  plugins: [],
};
