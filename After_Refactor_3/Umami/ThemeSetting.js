import { useState } from 'react';
import { Button } from 'react-basics';
import useTheme from 'hooks/useTheme';
import Sun from 'assets/sun.svg';
import Moon from 'assets/moon.svg';
import styles from './ThemeSetting.module.css';

const LIGHT_THEME = 'light';
const DARK_THEME = 'dark';

function ThemeButton({ isActive, onClick, children }) {
  return (
    <Button className={classNames({ [styles.active]: isActive })} onClick={onClick}>
      {children}
    </Button>
  );
}

function ThemeIcon({ icon }) {
  return (
    <Icon>
      {icon}
    </Icon>
  );
}

export function ThemeSetting() {
  const [theme, setTheme] = useTheme();
  const isLight = theme === LIGHT_THEME;
  const isDark = theme === DARK_THEME;

  const handleLightClick = () => setTheme(LIGHT_THEME);
  const handleDarkClick = () => setTheme(DARK_THEME);

  return (
    <div className={styles.buttons}>
      <ThemeButton isActive={isLight} onClick={handleLightClick}>
        <ThemeIcon icon={<Sun />} />
      </ThemeButton>
      <ThemeButton isActive={isDark} onClick={handleDarkClick}>
        <ThemeIcon icon={<Moon />} />
      </ThemeButton>
    </div>
  );
}

export default ThemeSetting;

