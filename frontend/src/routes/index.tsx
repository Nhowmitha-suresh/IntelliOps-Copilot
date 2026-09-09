import { createFileRoute } from "@tanstack/react-router";
import * as Collapsible from "@radix-ui/react-collapsible";
import * as Slider from "@radix-ui/react-slider";
import { Check, ChevronDown, Clock3, Copy, Cpu, History, Play, SlidersHorizontal, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { AmbientNetwork } from "@/components/ambient-network";
import { Button } from "@/components/ui/button";
import { generateText, generateInstruction, fetchModelInfo, fetchHealth, type ModelId } from "@/lib/model-api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [
    { title: "MiniGPT-Scratch — Transformer Lab" },
    { name: "description", content: "Test and compare a transformer language model built entirely from scratch." },
    { property: "og:title", content: "MiniGPT-Scratch — Transformer Lab" },
    { property: "og:description", content: "Test and compare a transformer language model built entirely from scratch." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ]}),
  component: Index,
});

const defaultModels = {
  baseline: { name: "Baseline", label: "GPT-S / 01", params: "875K", vocab: "512", perplexity: "11.77", blurb: "Original pre-trained architecture and run." },
  variant: { name: "Variant", label: "GPT-S / 02", params: "3.32M", vocab: "512", perplexity: "9.13", blurb: "Optimized depth with improved regularization." },
  sft: { name: "Instruction-Tuned (SFT)", label: "GPT-S / 03 SFT", params: "875K", vocab: "512", perplexity: "2.53", blurb: "Fine-tuned to follow Shakespearean instructions." },
};

function formatParams(count: number): string {
  if (count >= 1e6) {
    return `${(count / 1e6).toFixed(2)}M`;
  } else if (count >= 1e3) {
    return `${(count / 1e3).toFixed(0)}K`;
  }
  return count.toString();
}

type Result = { id: number; model: ModelId; prompt: string; text: string; time: string };

function Index() {
  const [selected, setSelected] = useState<ModelId>("sft");
  const [compare, setCompare] = useState(false);
  const [prompt, setPrompt] = useState("Write a short speech about courage");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(120);
  const [penalty, setPenalty] = useState(1.1);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Partial<Record<ModelId, Result>>>({});
  const [history, setHistory] = useState<Result[]>([]);
  const [modelStats, setModelStats] = useState(defaultModels);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const outputRef = useRef<HTMLElement>(null);

  const isInstructionMode = selected === "sft" || compare;

  useEffect(() => {
    fetchModelInfo()
      .then((info) => {
        setModelStats({
          baseline: {
            ...defaultModels.baseline,
            params: info.baseline ? formatParams(info.baseline.param_count) : defaultModels.baseline.params,
            vocab: info.baseline ? info.baseline.vocab_size.toLocaleString() : defaultModels.baseline.vocab,
            perplexity: info.baseline?.val_perplexity ? info.baseline.val_perplexity.toFixed(2) : defaultModels.baseline.perplexity,
          },
          variant: {
            ...defaultModels.variant,
            params: info.variant ? formatParams(info.variant.param_count) : defaultModels.variant.params,
            vocab: info.variant ? info.variant.vocab_size.toLocaleString() : defaultModels.variant.vocab,
            perplexity: info.variant?.val_perplexity ? info.variant.val_perplexity.toFixed(2) : defaultModels.variant.perplexity,
          },
          sft: {
            ...defaultModels.sft,
            params: info.sft ? formatParams(info.sft.param_count) : defaultModels.sft.params,
            vocab: info.sft ? info.sft.vocab_size.toLocaleString() : defaultModels.sft.vocab,
            perplexity: info.sft?.val_perplexity ? info.sft.val_perplexity.toFixed(2) : defaultModels.sft.perplexity,
          },
        });
      })
      .catch((err) => console.error("Could not fetch backend model info:", err));

    fetchHealth()
      .then((res) => {
        setIsHealthy(res.status === "ok" && res.baseline_loaded && res.variant_loaded);
      })
      .catch(() => setIsHealthy(false));
  }, []);

  async function handleGenerate() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    const targets: ModelId[] = compare ? ["baseline", "variant", "sft"] : [selected];
    try {
      const generated = await Promise.all(targets.map(async (model) => {
        let textOutput = "";
        if (model === "sft") {
          const res = await generateInstruction({
            instruction: prompt.trim(),
            max_new_tokens: maxTokens,
            temperature,
            repetition_penalty: penalty,
          });
          textOutput = res.generated_response;
        } else {
          const res = await generateText({
            prompt: prompt.trim(),
            model,
            max_new_tokens: maxTokens,
            temperature,
            repetition_penalty: penalty,
          });
          textOutput = res.generated_text;
        }

        return {
          id: Date.now() + (model === "variant" ? 1 : model === "sft" ? 2 : 0),
          model,
          prompt: prompt.trim(),
          text: textOutput,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        } satisfies Result;
      }));
      setResults(Object.fromEntries(generated.map((item) => [item.model, item])));
      setHistory((old) => [...generated, ...old].slice(0, 10));
    } catch (err: any) {
      console.error("Generation error:", err);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => outputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }));
    }
  }

  const continuationExamples = ["Once upon a time", "To be or not to be", "The King said"];
  const instructionExamples = [
    "Write a short speech about courage",
    "Write two lines about a stormy night",
    "Continue this: The castle stood silent",
  ];

  return (
    <main className="min-h-screen overflow-hidden bg-background text-foreground">
      <AmbientNetwork />
      <header className="mx-auto flex h-20 max-w-6xl items-center justify-between px-5 sm:px-8">
        <a href="#top" className="flex items-center gap-3 font-mono text-xs tracking-widest text-muted-foreground"><span className="grid size-8 place-items-center rounded-md border border-primary/40 bg-primary/10 text-primary"><Cpu size={16}/></span> MINIGPT / LAB</a>
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          <span className={cn("size-1.5 rounded-full", isHealthy === true ? "bg-teal animate-pulse" : isHealthy === false ? "bg-destructive" : "bg-muted")} />
          {isHealthy === true ? "Backend: Online" : isHealthy === false ? "Backend: Offline" : "Connecting..."}
        </div>
      </header>

      <div id="top" className="mx-auto max-w-6xl px-5 pb-24 pt-14 sm:px-8 sm:pt-20">
        <section className="max-w-4xl animate-[reveal_.7s_ease-out]">
          <div className="mb-5 flex items-center gap-2 font-mono text-xs uppercase tracking-[.2em] text-primary"><Sparkles size={14}/> Neural language research</div>
          <h1 className="text-5xl font-semibold leading-none tracking-normal sm:text-7xl lg:text-8xl">MiniGPT-<span className="bg-gradient-action bg-clip-text text-transparent">Scratch</span></h1>
          <p className="mt-7 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">A transformer language model built and trained entirely from scratch — featuring pretraining and SFT instruction fine-tuning.</p>
          <div className="mt-9 flex flex-wrap gap-x-8 gap-y-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground"><span>Decoder-only</span><span>BPE Tokenizer</span><span>Instruction Tuning</span><span>PyTorch</span></div>
        </section>

        <section className="mt-24" aria-labelledby="model-heading">
          <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
            <div><p className="font-mono text-xs uppercase tracking-[.18em] text-primary">01 / Model selection</p><h2 id="model-heading" className="mt-2 text-2xl font-semibold">Choose a model mode</h2></div>
            <label className="flex cursor-pointer items-center gap-3 text-sm text-muted-foreground"><span>Compare all 3</span><input type="checkbox" checked={compare} onChange={(e) => { const next = e.target.checked; setCompare(next); if (next && !instructionExamples.includes(prompt)) setPrompt("Write a short speech about courage"); }} className="peer sr-only"/><span className="relative h-6 w-11 rounded-full border border-border bg-muted transition peer-checked:border-primary/60 peer-checked:bg-primary/20 after:absolute after:left-1 after:top-1 after:size-3.5 after:rounded-full after:bg-muted-foreground after:transition-all peer-checked:after:translate-x-5 peer-checked:after:bg-primary"/></label>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {(["baseline", "variant", "sft"] as ModelId[]).map((id) => (
              <ModelCard
                key={id}
                id={id}
                stats={modelStats[id]}
                selected={compare || selected === id}
                onSelect={() => {
                  setSelected(id);
                  setCompare(false);
                  if (id === "sft") setPrompt("Write a short speech about courage");
                }}
              />
            ))}
          </div>
        </section>

        <section className="mt-12 border-y border-border bg-surface/60 px-5 py-7 backdrop-blur-sm sm:px-8 sm:py-9" aria-labelledby="prompt-heading">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="font-mono text-xs uppercase tracking-[.18em] text-primary">
                02 / Inference ({isInstructionMode ? "Instruction Mode" : "Continuation Mode"})
              </p>
              <h2 id="prompt-heading" className="mt-2 text-xl font-semibold">
                {isInstructionMode ? "Enter an instruction" : "Enter a prompt"}
              </h2>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{prompt.length} chars</span>
          </div>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder={isInstructionMode ? "Write a short monologue about honor..." : "Write the beginning of a story…"}
            className="w-full resize-none border-0 bg-transparent p-0 text-xl leading-8 text-foreground outline-none placeholder:text-muted-foreground/50 sm:text-2xl"
          />

          <div className="mt-5 flex flex-wrap gap-2">
            {(isInstructionMode ? instructionExamples : continuationExamples).map((example) => (
              <button
                key={example}
                onClick={() => setPrompt(example)}
                className="rounded-full border border-border px-3 py-1.5 font-mono text-[10px] text-muted-foreground transition hover:border-primary/60 hover:text-primary"
              >
                {example}
              </button>
            ))}
          </div>

          {isInstructionMode && (
            <div className="mt-5 rounded-md border border-primary/30 bg-primary/10 p-3.5 text-xs leading-5 text-muted-foreground">
              <span className="font-semibold text-primary">Note on SFT Mode:</span> Fine-tuned to follow instructions in Shakespeare's style — learned to respond in character-dialogue format, though semantic understanding is limited by the small fine-tuning dataset (150 examples).
            </div>
          )}

          <Settings temperature={temperature} setTemperature={setTemperature} maxTokens={maxTokens} setMaxTokens={setMaxTokens} penalty={penalty} setPenalty={setPenalty}/>
          <div className="mt-7 flex justify-end">
            <Button variant="gradient" onClick={handleGenerate} disabled={!prompt.trim() || loading} className="h-12 min-w-40">
              <Play size={15} fill="currentColor"/>{loading ? "Generating…" : compare ? "Compare all 3" : isInstructionMode ? "Follow instruction" : "Generate text"}
            </Button>
          </div>
        </section>

        <section ref={outputRef} className="mt-12" aria-labelledby="output-heading">
          <div className="mb-5 flex items-end justify-between"><div><p className="font-mono text-xs uppercase tracking-[.18em] text-primary">03 / Output</p><h2 id="output-heading" className="mt-2 text-xl font-semibold">Model response</h2></div>{loading && <LoadingNodes/>}</div>
          <div className={cn("grid gap-4", compare ? "lg:grid-cols-3 md:grid-cols-2" : "")}>
            {loading ? (compare ? ["baseline", "variant", "sft"] as ModelId[] : [selected]).map((id) => <OutputSkeleton key={id} model={id}/>) :
              Object.values(results).length ? Object.values(results).map((result) => result && <OutputPanel key={result.id} result={result}/>) : <div className="grid min-h-56 place-items-center border border-dashed border-border bg-surface/40 text-center"><div><Sparkles className="mx-auto mb-3 text-muted-foreground" size={20}/><p className="text-sm text-muted-foreground">Generated text will appear here.</p></div></div>}
          </div>
        </section>

        <HistoryPanel history={history} onSelect={(item) => { setResults({ [item.model]: item }); setCompare(false); setSelected(item.model); }}/>
        <HowItWorks />
      </div>
      <footer className="border-t border-border px-5 py-8"><div className="mx-auto flex max-w-6xl flex-col justify-between gap-3 font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:flex-row"><span>MiniGPT-Scratch / Capstone research project</span><span>Built from first principles</span></div></footer>
    </main>
  );
}

function ModelCard({ id, stats, selected, onSelect }: { id: ModelId; stats: { name: string; label: string; params: string; vocab: string; perplexity: string; blurb: string }; selected: boolean; onSelect: () => void }) {
  return <button onClick={onSelect} aria-pressed={selected} className={cn("group relative min-h-64 overflow-hidden border bg-card/75 p-6 text-left backdrop-blur-sm transition duration-300 hover:-translate-y-1 hover:border-primary/50", selected ? "border-primary/70 shadow-[0_0_35px_color-mix(in_oklab,var(--teal)_12%,transparent)] animate-[border-flow_4s_ease-in-out_infinite]" : "border-border")}>
    <div className="flex items-start justify-between"><span className="font-mono text-[10px] uppercase tracking-[.2em] text-muted-foreground">{stats.label}</span><span className={cn("grid size-6 place-items-center rounded-full border transition", selected ? "border-primary bg-primary text-primary-foreground" : "border-border text-transparent")}><Check size={13}/></span></div>
    <h3 className="mt-7 text-2xl font-semibold">{stats.name}</h3><p className="mt-2 text-sm text-muted-foreground">{stats.blurb}</p>
    <dl className="mt-8 grid grid-cols-3 gap-3 border-t border-border pt-5">{[["Parameters",stats.params],["Vocabulary",stats.vocab],["Val. PPL",stats.perplexity]].map(([label,value]) => <div key={label}><dt className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">{label}</dt><dd className="mt-2 font-mono text-base text-foreground">{value}</dd></div>)}</dl>
  </button>;
}

function Settings(props: { temperature:number; setTemperature:(v:number)=>void; maxTokens:number; setMaxTokens:(v:number)=>void; penalty:number; setPenalty:(v:number)=>void }) {
  return <Collapsible.Root className="mt-6 border-t border-border pt-5"><Collapsible.Trigger className="group flex w-full items-center gap-2 text-sm text-muted-foreground transition hover:text-foreground"><SlidersHorizontal size={14}/> Advanced settings <ChevronDown size={14} className="ml-auto transition group-data-[state=open]:rotate-180"/></Collapsible.Trigger><Collapsible.Content className="mt-6 grid gap-7 data-[state=open]:animate-accordion-down md:grid-cols-3"><Control label="Temperature" value={props.temperature} min={0.1} max={1.5} step={0.1} onChange={props.setTemperature}/><Control label="Max tokens" value={props.maxTokens} min={10} max={500} step={10} onChange={props.setMaxTokens}/><Control label="Repetition penalty" value={props.penalty} min={1} max={2} step={0.1} onChange={props.setPenalty}/></Collapsible.Content></Collapsible.Root>;
}

function Control({ label, value, min, max, step, onChange }: { label:string; value:number; min:number; max:number; step:number; onChange:(v:number)=>void }) {
  return <div><div className="mb-3 flex justify-between text-xs"><span className="text-muted-foreground">{label}</span><span className="font-mono text-primary">{value.toFixed(step < 1 ? 1 : 0)}</span></div><Slider.Root min={min} max={max} step={step} value={[value]} onValueChange={([v]) => v !== undefined && onChange(v)} className="relative flex h-5 touch-none items-center"><Slider.Track className="relative h-1 grow overflow-hidden rounded-full bg-muted"><Slider.Range className="absolute h-full bg-gradient-action"/></Slider.Track><Slider.Thumb className="block size-4 rounded-full border-2 border-primary bg-background outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"/></Slider.Root></div>;
}

function LoadingNodes() { return <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-primary"><span>Thinking</span><span className="flex gap-1">{[0,1,2].map(i => <i key={i} className="size-1 rounded-full bg-primary" style={{ animation: `pulse-node 1s ease-in-out ${i*.16}s infinite` }}/>)}</span></div> }
function OutputSkeleton({model}:{model:ModelId}) { return <div className="min-h-64 border border-border bg-card/60 p-6"><div className="font-mono text-[10px] uppercase tracking-widest text-primary">{defaultModels[model].name}</div><div className="mt-8 space-y-3">{["w-full","w-11/12","w-4/5","w-2/3"].map((w,i)=><div key={i} className={cn("h-2 animate-pulse rounded bg-muted",w)}/>)}</div></div> }
function OutputPanel({result}:{result:Result}) { const [copied,setCopied]=useState(false); return <article className="min-h-64 border border-border bg-card/75 p-6 backdrop-blur-sm animate-[reveal_.5s_ease-out]"><header className="flex justify-between"><span className="font-mono text-[10px] uppercase tracking-[.18em] text-primary">{defaultModels[result.model].name.toUpperCase()} / GENERATED</span><Button size="icon" variant="ghost" aria-label="Copy generated text" title="Copy generated text" onClick={() => { navigator.clipboard.writeText(result.text); setCopied(true); setTimeout(()=>setCopied(false),1200); }}>{copied ? <Check size={15}/> : <Copy size={15}/>}</Button></header><p className="mt-6 whitespace-pre-wrap text-base leading-8 text-foreground/90">{result.text}</p></article> }

function HistoryPanel({history,onSelect}:{history:Result[];onSelect:(item:Result)=>void}) { return <Collapsible.Root className="mt-12 border-t border-border pt-6"><Collapsible.Trigger className="group flex w-full items-center gap-3 text-left"><History size={17} className="text-primary"/><span className="font-medium">Generation history</span><span className="font-mono text-[10px] text-muted-foreground">{history.length} SESSION ITEMS</span><ChevronDown size={15} className="ml-auto text-muted-foreground transition group-data-[state=open]:rotate-180"/></Collapsible.Trigger><Collapsible.Content className="mt-5 grid gap-2 data-[state=open]:animate-accordion-down">{history.length === 0 ? <p className="py-5 text-sm text-muted-foreground">Your recent generations will be kept here for this session.</p> : history.map(item=><button key={item.id} onClick={()=>onSelect(item)} className="flex items-center gap-4 border border-border bg-surface/50 p-3 text-left transition hover:border-primary/40"><span className="rounded border border-border px-2 py-1 font-mono text-[9px] uppercase text-primary">{item.model}</span><span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">{item.prompt}</span><span className="flex items-center gap-1 font-mono text-[9px] text-muted-foreground"><Clock3 size={11}/>{item.time}</span></button>)}</Collapsible.Content></Collapsible.Root> }

function HowItWorks() { return <Collapsible.Root className="mt-16 border border-border bg-surface/50"><Collapsible.Trigger className="group flex w-full items-center p-5 text-left sm:p-7"><div><p className="font-mono text-[10px] uppercase tracking-[.18em] text-primary">Under the hood</p><h2 className="mt-2 text-xl font-semibold">How it works</h2></div><ChevronDown className="ml-auto text-muted-foreground transition group-data-[state=open]:rotate-180"/></Collapsible.Trigger><Collapsible.Content className="px-5 pb-7 data-[state=open]:animate-accordion-down sm:px-7"><div className="grid gap-8 border-t border-border pt-7 md:grid-cols-3">{[["01","Tokenize","Your words are broken into small pieces the model can understand."],["02","Transform","Layers of attention look for patterns and relationships in the sequence."],["03","Generate","The model predicts one new piece at a time, building the response as it goes."]].map(([n,title,body])=><div key={n}><span className="font-mono text-xs text-primary">{n}</span><h3 className="mt-3 font-semibold">{title}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{body}</p></div>)}</div></Collapsible.Content></Collapsible.Root> }