---
layout: page
title: About
permalink: /about
---

## Personal Information

**Address:** Kortrijk, Belgium  
**Email:** <a href="mailto:&#105;&#98;&#114;&#97;&#104;&#105;&#109;&#95;&#101;&#108;&#107;&#97;&#100;&#100;&#111;&#117;&#114;&#105;&#64;&#111;&#117;&#116;&#108;&#111;&#111;&#107;&#46;&#99;&#111;&#109;">Ibrahim El Kaddouri</a>

<a href="https://github.com/IbrahimElk">
  <img src="/assets/social/github.png" alt="GitHub" width="48" height="48">
</a>
<a href="https://linkedin.com/in/ibrahimelk/">
  <img src="/assets/social/linkedin.png" alt="Linkedin" width="48" height="48">
</a>
<a href="https://bit.ly/4fPyhMM">
  <img src="/assets/social/noot.jpg" alt="noot" width="48" height="48">
</a>

<br>

## Employment History

### Student Software Engineer - KRR group at KU Leuven

- The development of [IDP-Z3](https://www.idp-z3.be/) reasoning application for the
  Knowledge Representation and Reasoning group
- Worked mainly on the Interactive Consultant and
  [Web IDE](https://interactive-consultant.idp-z3.be/)
  part of the application, see the
  [changelog](https://gitlab.com/krr/IDP-Z3/-/blob/9a3362df5474cde981d2fde574a738b4019fa4b4/CHANGELOG.md#0112---18-10-2024)

### Research Intern - Technische Universität Graz

- The internship revolves around designing [empathetic agents](projects/2024/09/05/empathicAgents.html)
  as conversation partners for people with autism spectrum disorder.

### Observatory Intern - Royal Observatory of Belgium

- Attended [lectures](activities/2024/02/05/observatory.html) at the [observatory](https://www.astro.oma.be/en/)
  on topics such as:
  reference systems and planetology, solar physics and space weather,
  astronomy and astrophysics and seismology and gravimetry.

### Student Software Engineer - Diggieteam Brussel

- [Linked data](https://rubenverborgh.github.io/WebFundamentals/semantic-web/) and semantic web
- Implemented SPARQL queries for an RDF graph database

### Student Firmware Engineer - Crestron Mechelen

- It primarily revolved around controlling and updating
  the firmware of Crestron devices.

<br>

## Open Source Contributions

### MyoFInDer - Myoblast Fusion Index Determination Software

- Contributed to [MyoFInDer](https://github.com/TissueEngineeringLab),
  see [this](projects/2021/11/22/cellenTellen.html)
  interesting piece of work
- Played with a forked
  [MyoFInDer](https://github.com/IbrahimElk/MyoFInDer)
- Explored the Tauri framework
- Seperated the GUI from the Python backend
- Tried to reduce program's storage footprint on the client side

<br>

## Skills

- **Preferred Programming Languages:** Python, C++/C, TypeScript
- **Prefered Env:**
  Git, Vim, Tmux, Zsh, Kitty, Hyperland, <a href="https://github.com/IbrahimElk/dotfiles/">Arch</a>, Docker

  ![languages](/assets/social/languages.svg)

<br>

## Education

- **Master of Computer Science:** Ku Leuven Universiteit, 2025
- **Bachelor of Computer Science:** Ku Leuven Universiteit, 2023
- **Languages:** English (fluent), Dutch (fluent), French (mediocre)

<div id="shape">
  <style>
    .hidden {
    }
    #output2 {
      background: #fff;
      overflow-x: overflow;
      transform: translateX(-20%);
      overflow-y: overflow;
      height: 100vh;
    }
    #output2 pre {
      border-left: none;
      overflow: visible;
    }
  </style>
  <div id="output2">
    <!-- ASCII art will be inserted here -->
  </div>
  <script type="module">
    import { fileNames, loadFileConcurrently, play } from '/assets/ts/easter_egg.js';
    async function main() {
      const fileName = fileNames[1];
      const content = await loadFileConcurrently(fileName);
      if (content != undefined) {
        await play(content, "output2");
      }
    }
    main();
  </script>
</div>
