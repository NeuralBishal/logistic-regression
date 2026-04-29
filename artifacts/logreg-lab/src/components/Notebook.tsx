import { type ReactNode } from "react";
import { Lightbulb, Code as CodeIcon, FunctionSquare } from "lucide-react";

export function Cell({
  kind,
  title,
  children,
}: {
  kind: "explain" | "math" | "code" | "viz";
  title?: string;
  children: ReactNode;
}) {
  const meta: Record<
    "explain" | "math" | "code" | "viz",
    { label: string; icon: ReactNode; tone: string }
  > = {
    explain: {
      label: "Explanation",
      icon: <Lightbulb className="w-3.5 h-3.5" />,
      tone: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-100",
    },
    math: {
      label: "Math",
      icon: <FunctionSquare className="w-3.5 h-3.5" />,
      tone: "bg-violet-100 text-violet-900 dark:bg-violet-900/40 dark:text-violet-100",
    },
    code: {
      label: "Code",
      icon: <CodeIcon className="w-3.5 h-3.5" />,
      tone: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-100",
    },
    viz: {
      label: "Visualization",
      icon: <CodeIcon className="w-3.5 h-3.5" />,
      tone: "bg-sky-100 text-sky-900 dark:bg-sky-900/40 dark:text-sky-100",
    },
  };
  const m = meta[kind];

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-muted/40">
        <span
          className={`inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded ${m.tone}`}
        >
          {m.icon}
          {m.label}
        </span>
        {title && (
          <span className="text-sm font-medium text-foreground">{title}</span>
        )}
      </div>
      <div className="p-4 text-sm leading-relaxed">{children}</div>
    </div>
  );
}

export function CodeBlock({ children }: { children: string }) {
  // Very lightweight syntax highlighting via classes
  const lines = children.split("\n");
  return (
    <pre className="code-cell">
      {lines.map((line, i) => (
        <div key={i}>
          <span
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: highlight(line) }}
          />
        </div>
      ))}
    </pre>
  );
}

function highlight(line: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let out = escape(line);
  out = out.replace(/(#.*$)/g, '<span class="cm">$1</span>');
  out = out.replace(
    /\b(import|from|def|return|for|in|if|else|elif|class|as|with|while|lambda|None|True|False)\b/g,
    '<span class="kw">$1</span>',
  );
  out = out.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="num">$1</span>');
  out = out.replace(/("[^"]*"|'[^']*')/g, '<span class="str">$1</span>');
  out = out.replace(
    /\b([a-zA-Z_][\w]*)(?=\()/g,
    '<span class="fn">$1</span>',
  );
  return out;
}
