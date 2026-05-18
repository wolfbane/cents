// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://dollars-and-cents.ai",
  integrations: [
    starlight({
      title: "Cents",
      description:
        "Thesis-driven investment research, agent-orchestrated. CLI documentation for the cents tool.",
      customCss: ["./src/styles/custom.css"],
      components: {
        Footer: "./src/components/Footer.astro",
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/wolfbane/cents",
        },
      ],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Overview", link: "/" },
            { label: "Scope (read first)", link: "/scope/", badge: { text: "Important", variant: "caution" } },
            { label: "Operating principles", link: "/principles/" },
            { label: "Quickstart", link: "/quickstart/" },
            { label: "Configuration", link: "/configuration/" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "The factory", link: "/factory/" },
            { label: "Screeners", link: "/screeners/" },
            { label: "Universes", link: "/universe/" },
            { label: "Cohorts", link: "/cohort/" },
            { label: "Events & invalidation", link: "/events/" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Commands", link: "/commands/" },
            { label: "Agents", link: "/agents/" },
            { label: "Architecture", link: "/architecture/" },
          ],
        },
        {
          label: "Research workflow",
          items: [
            { label: "cents experiment", link: "/commands/experiment/" },
            { label: "cents calibration", link: "/commands/calibration/" },
            { label: "cents eval", link: "/commands/eval/" },
            { label: "cents evidence", link: "/commands/evidence/" },
            { label: "cents shadow", link: "/commands/shadow/" },
          ],
        },
        {
          label: "Project",
          items: [{ label: "Roadmap", link: "/roadmap/" }],
        },
      ],
    }),
  ],
});
