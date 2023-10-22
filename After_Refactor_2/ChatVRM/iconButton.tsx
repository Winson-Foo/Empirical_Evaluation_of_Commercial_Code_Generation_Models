import { KnownIconType } from "@charcoal-ui/icons";
import { ButtonHTMLAttributes } from "react";

const PRIMARY_BUTTON_STYLES = {
  bg: "bg-primary",
  hoverBg: "hover:bg-primary-hover",
  activeBg: "active:bg-primary-press",
  disabledBg: "disabled:bg-primary-disabled",
  text: "text-white",
  rounded: "rounded-16",
  size: "text-sm p-8",
  align: "text-center inline-flex items-center",
  margin: "mr-2",
};

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  iconName: keyof KnownIconType;
  isProcessing: boolean;
  label?: string;
};

const IconButton = ({ iconName, isProcessing, label, ...rest }: Props) => {
  const renderIcon = () => {
    const icon = isProcessing ? "24/Dot" : iconName;
    return <pixiv-icon name={icon} scale="1"></pixiv-icon>;
  };

  const renderLabel = () => {
    return label ? (
      <div className="mx-4 font-M_PLUS_2 font-bold">{label}</div>
    ) : null;
  };

  return (
    <button
      {...rest}
      className={`${PRIMARY_BUTTON_STYLES.bg} ${
        PRIMARY_BUTTON_STYLES.hoverBg
      } ${PRIMARY_BUTTON_STYLES.activeBg} ${
        PRIMARY_BUTTON_STYLES.disabledBg
      } ${PRIMARY_BUTTON_STYLES.text} ${PRIMARY_BUTTON_STYLES.rounded} ${
        PRIMARY_BUTTON_STYLES.size
      } ${PRIMARY_BUTTON_STYLES.align} ${
        PRIMARY_BUTTON_STYLES.margin
      } ${rest.className}`}
    >
      {renderIcon()}
      {renderLabel()}
    </button>
  );
};

export default IconButton;

