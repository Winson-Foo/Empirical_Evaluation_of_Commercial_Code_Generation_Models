import React from 'react';
import { Icon, Button, PopupTrigger, Popup, Text } from 'react-basics';
import classNames from 'classnames';
import { languages } from 'lib/lang';
import useLocale from 'hooks/useLocale';
import Icons from 'components/icons';
import styles from './LanguageButton.module.css';

function LanguageSelector() {
  const { locale, saveLocale, dir } = useLocale();
  const items = getItems();

  function getItems() {
    return Object.keys(languages).map(key => ({
      label: languages[key].label,
      value: key,
    }));
  }

  function handleSelect(value) {
    saveLocale(value);
  }

  return (
    <PopupTrigger>
      <Button variant="quiet">
        <Icon>
          <Icons.Globe />
        </Icon>
      </Button>
      <Popup position="bottom" alignment={dir === 'rtl' ? 'start' : 'end'}>
        <div className={styles.menu}>
          {items.map(({ value, label }) => (
            <div
              key={value}
              className={classNames(styles.item, { [styles.selected]: value === locale })}
              onClick={() => handleSelect(value)}
            >
              <Text>{label}</Text>
              {value === locale && (
                <Icon className={styles.icon}>
                  <Icons.Check />
                </Icon>
              )}
            </div>
          ))}
        </div>
      </Popup>
    </PopupTrigger>
  );
}

export default LanguageSelector;

