import { useState } from 'react';
import { ThemeProvider } from 'hooks/useTheme';
import styles from './ThemeSetting.module.css';

export function ThemeSetting() {
  const [theme, setTheme] = useState('light');

  return (
    <ThemeProvider value={{ theme, setTheme }}>
      <div className={styles.buttons}>
        <ThemeButton label="Light" icon={<Sun />} value="light" />
        <ThemeButton label="Dark" icon={<Moon />} value="dark" />
      </div>
    </ThemeProvider>
  );
}

function ThemeButton({ label, icon, value }) {
  const { theme, setTheme } = useTheme();

  return (
    <button
      className={classNames(styles.button, {
        [styles.active]: theme === value,
      })}
      onClick={() => setTheme(value)}
      aria-label={`Switch to ${label} theme`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export default ThemeSetting;