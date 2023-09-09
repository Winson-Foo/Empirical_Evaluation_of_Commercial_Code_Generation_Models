import React from 'react';
import { Button, Icon, Text } from 'components';
import classNames from 'classnames';
import { languages } from 'lib/lang';
import useLocale from 'hooks/useLocale';
import Icons from 'components/icons';
import styles from './LanguageButton.module.css';

function LanguageItem({ value, label, isSelected, onSelect }) {
  const itemClasses = classNames(styles.item, { [styles.selected]: isSelected });
  const handleSelect = () => onSelect(value);

  return (
    <div className={itemClasses} onClick={handleSelect}>
      <Text>{label}</Text>
      {isSelected && (
        <Icon className={styles.icon}>
          <Icons.Check />
        </Icon>
      )}
    </div>
  );
}

function LanguageMenu({ items, selectedValue, onSelect }) {
  return (
    <div className={styles.menu}>
      {items.map((item) => (
        <LanguageItem
          key={item.value}
          value={item.value}
          label={item.label}
          isSelected={item.value === selectedValue}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

function LanguageButton() {
  const { locale, saveLocale, dir } = useLocale();
  const items = Object.keys(languages).map((key) => ({ ...languages[key], value: key }));
  const handleSelect = (value) => saveLocale(value);

  return (
    <Button variant="quiet" icon={<Icons.Globe />}>
      <LanguageMenu items={items} selectedValue={locale} onSelect={handleSelect} />
    </Button>
  );
}

export default LanguageButton;