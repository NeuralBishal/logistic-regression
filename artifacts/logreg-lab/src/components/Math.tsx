import katex from "katex";
import { useMemo } from "react";

interface Props {
  tex: string;
  display?: boolean;
  className?: string;
}

export function Math({ tex, display = false, className }: Props) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        throwOnError: false,
        displayMode: display,
        output: "html",
      });
    } catch {
      return tex;
    }
  }, [tex, display]);
  return (
    <span
      className={className}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
