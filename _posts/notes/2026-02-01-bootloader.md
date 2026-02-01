---
layout: post
title: "Booting Linux From Scratch"
date:  2026-02-01
keywords: ["raspberry-pi","bootloader","qemu","lfs","bare-metal","toolchain"]
categories: notes
---
<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD003 -->

**Author:** Ibrahim El Kaddouri  

**Repository:** <span class="tooltip">
<a href="https://github.com/IbrahimElk/baremetal-to-lfs">
    <img src="/assets/images/2026-02-01/icon.jpg" style="width: 5%">
</a>
<span class="tooltip-text"> The repository is private </span>
</span>
<br>
<br>

## Coming Soon

<br>
This project aims to design and implement a minimal, reproducible Linux boot
pipeline for the Raspberry Pi, developed primarily under emulation (QEMU) and
validated on real hardware. The work is split into two main deliverables:

* a custom second-stage bootloader capable of loading a Linux kernel and device tree
* a complete Linux From Scratch (LFS) userland booting on top of that pipeline.

A key constraint is that the project will use QBE as the code-generation backend
for compilation experiments and for producing low-level binaries used in the early
boot components

## References

1. [baking pi](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/ok01.html)  
2. [qbe](https://c9x.me/compile/)  
3. [qbe blog](https://briancallahan.net/blog/20210829.html)  
4. [linker](https://lwn.net/Articles/276782/)  
5. [risc OS](https://www.riscosopen.org/tracker/)  
6. [BMOW](https://www.bigmessowires.com/bmow1/)  
7. [mirroprocessor](https://en.wikibooks.org/wiki/Microprocessor_Design)  
8. [machinecode](http://www.goodmath.org/blog/2014/01/02/leading-in-to-machine-code-why/)  
