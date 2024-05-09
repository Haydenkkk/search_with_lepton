import { Mails } from "lucide-react";
import { FC } from "react";

export const Footer: FC = () => {
  return (
    <div className="text-center flex flex-col items-center text-xs text-zinc-700 gap-1">
      <div className="text-zinc-400">
        利用本地知识库由大型语言模型生成的答案，正确性请具体验证。
      </div>
      {/* <div className="text-zinc-400">
        LLM, Vector DB, and other components powered by the Lepton AI platform.
      </div> */}
      {/* <div className="flex gap-2 justify-center">
        <div>如果您在使用中遇到任何bug，</div>
        <div>
          <a
            className="text-blue-500 font-medium inline-flex gap-1 items-center flex-nowrap text-nowrap"
            href="mailto:lihaoixin@bupr.edu.cn"
          >
            <Mails size={8} />
            请联系我们....
          </a>
        </div>

      </div> */}

      {/* <div className="flex items-center justify-center flex-wrap gap-x-4 gap-y-2 mt-2 text-zinc-400">
        <a className="hover:text-zinc-950" href="https://lepton.ai/">
          Lepton Home
        </a>
        <a
          className="hover:text-zinc-950"
          href="https://dashboard.lepton.ai/playground"
        >
          API Playground
        </a>
        <a
          className="hover:text-zinc-950"
          href="https://github.com/leptonai/leptonai"
        >
          Python Library
        </a>
        <a className="hover:text-zinc-950" href="https://twitter.com/leptonai">
          Twitter
        </a>
        <a className="hover:text-zinc-950" href="https://leptonai.medium.com/">
          Blog
        </a>
      </div> */}
    </div>
  );
};
