// languages.js
export const languages = {
  en: { label: 'English', dir: 'ltr' },
  es: { label: 'Espa�ol', dir: 'ltr' },
  // add more languages here
};

// PopupTrigger.js
import { PopupTrigger as BasicsPopupTrigger } from 'react-basics';

export default BasicsPopupTrigger;

// Button.js
import { Button as BasicsButton } from 'react-basics';

export default BasicsButton;

// Icon.js
import { Icon as BasicsIcon } from 'react-basics';

export default BasicsIcon;

// Popup.js
import { Popup as BasicsPopup } from 'react-basics';

export default BasicsPopup;

// Text.js
import { Text as BasicsText } from 'react-basics';

export default BasicsText;

import { Icon, Button, PopupTrigger, Popup, Text } from 'react-basics';
import classNames from 'classnames';
import { languages } from 'lib/lang';
import useLocale from 'hooks/useLocale';
import Icons from 'components/icons';
import styles from './LanguageSelector.module.css';

export function LanguageSelector() {
  const { locale, saveLocale, dir } = useLocale();
  const { menu, item, selected, icon } = styles;
  const items = Object.entries(languages).map(([key, { label }]) => ({ value: key, label }));

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
        <div className={menu}>
          {items.map(({ value, label }) => {
            return (
              <div
                key={value}
                className={classNames(item, { [selected]: value === locale })}
                onClick={() => handleSelect(value)}
              >
                <Text>{label}</Text>
                {value === locale && (
                  <Icon className={icon}>
                    <Icons.Check />
                  </Icon>
                )}
              </div>
            );
          })}
        </div>
      </Popup>
    </PopupTrigger>
  );
}

export default LanguageSelector;