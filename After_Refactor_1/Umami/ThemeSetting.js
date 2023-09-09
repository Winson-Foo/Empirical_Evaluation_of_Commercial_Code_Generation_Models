import classNames from 'classnames';
import { Button, Icon } from 'react-basics';
import useTheme from 'hooks/useTheme';
import Sun from 'assets/sun.svg';
import Moon from 'assets/moon.svg';
import styles from './ThemeSetting.module.css';

function ThemeSetting() {
  const [theme, setTheme] = useTheme();

  function handleLightThemeClick() {
    setTheme('light');
  }

  function handleDarkThemeClick() {
    setTheme('dark');
  }

  const lightThemeButtonProps = {
    className: classNames({ [styles.active]: theme === 'light' }),
    onClick: handleLightThemeClick,
  };

  const darkThemeButtonProps = {
    className: classNames({ [styles.active]: theme === 'dark' }),
    onClick: handleDarkThemeClick,
  };

  return (
    <div className={styles.buttons}>
      <Button {...lightThemeButtonProps}>
        <Icon>
          <Sun />
        </Icon>
      </Button>
      <Button {...darkThemeButtonProps}>
        <Icon>
          <Moon />
        </Icon>
      </Button>
    </div>
  );
}

export default ThemeSetting;

