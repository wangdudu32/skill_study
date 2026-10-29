<script setup>
import CodeExample from '../../components/CodeExample.vue'
import ManualChecklist from '../../components/ManualChecklist.vue'
import CounterDemo from './CounterDemo.vue'
import CounterPractice from './CounterPractice.vue'
import CounterSolution from './solutions/CounterSolution.vue'
import counterSource from './CounterDemo.vue?raw'
import practiceSource from './CounterPractice.vue?raw'
import solutionSource from './solutions/CounterSolution.vue?raw'

defineEmits(['navigate'])

const checks = [
  { id: 'step', action: '重置，再连续点击两次加法', expected: '步长为 2，按钮显示 + 2 / − 2，计数依次为 2、4。' },
  { id: 'change-step', action: '接着选择步长 3，再点击一次加法', expected: '按钮文字变成 + 3 / − 3，计数从 4 变为 7。' },
  { id: 'upper', action: '重置，选择步长 3，加三次；再选步长 2，加一次', expected: '计数从 9 停在 10，加法按钮禁用；减一次后为 8，加法重新可用。' },
  { id: 'lower', action: '重置，选步长 1 加一次；改选步长 2，再减一次', expected: '计数从 1 变为 0，不能出现 -1；减法按钮禁用。' },
  { id: 'reset', action: '选择步长 3，加一次，再点击重置', expected: '计数恢复为 0，步长恢复为 2，按钮文字同步恢复。' },
]
</script>

<template>
  <section class="lesson-intro" aria-labelledby="lesson-title">
    <div class="eyebrow"><span class="pill">实验 01</span><span>基础入门 · 建议 25 分钟</span></div>
    <h1 id="lesson-title">从一个计数器，<br class="mobile-break" />理解<span>响应式。</span></h1>
    <p class="intro-description">改变一个数据，页面就跟着改变。亲手点几下，看看 Vue 是如何把它们连接起来的。</p>
    <div class="concept-tags" aria-label="本节知识点">
      <span><code>ref</code> 响应式状态</span>
      <span><code v-pre>{{ }}</code> 文本插值</span>
      <span><code>@click</code> 事件绑定</span>
      <span><code>:disabled</code> 属性绑定</span>
    </div>
  </section>
  <nav class="section-nav" aria-label="本节内容">
    <a href="#demo">01 体验</a>
    <a href="#source">02 读源码</a>
    <a href="#practice">03 写功能</a>
    <a href="#checklist">04 验证</a>
    <a href="#hints">05 提示</a>
  </nav>

  <section id="demo" class="lesson-section" aria-labelledby="demo-title">
    <div class="section-heading">
      <div><span class="section-kicker">PLAYGROUND</span><h2 id="demo-title">先点点看</h2></div>
      <span class="section-caption">操作左侧，观察数据变化</span>
    </div>
    <div class="demo-grid">
      <div class="demo-panel">
        <div class="panel-heading"><span><span class="status-dot"></span>参考演示 · 基础计数器</span><span>DEMO</span></div>
        <CounterDemo />
      </div>
      <aside class="observation-panel" aria-label="观察提示">
        <span class="small-label">带着问题去观察</span>
        <h3>你改的是数据，<br />Vue 更新的是页面。</h3>
        <ol class="observation-list">
          <li><strong>点一下「+ 1」</strong><p>大数字和下方的状态值一起改变，它们读取的是同一个 <code>count</code>。</p></li>
          <li><strong>把数字减到 0</strong><p>减法按钮自动禁用，因为 <code>count === 0</code> 变成了 <code>true</code>。</p></li>
          <li><strong>试试「重置为 0」</strong><p>只需要修改数据，相关的文字和按钮状态都会随之更新。</p></li>
        </ol>
        <p class="observation-note">思考：代码中有没有手动修改页面文字？</p>
      </aside>
    </div>
  </section>

  <section id="source" class="lesson-section" aria-labelledby="source-title">
    <div class="section-heading">
      <div><span class="section-kicker">UNDER THE HOOD</span><h2 id="source-title">把变化和代码对上号</h2></div>
      <span class="section-caption">先读逻辑，再看模板</span>
    </div>
    <div class="reading-guide">
      <strong>先读这个文件：<code>src/lessons/01-counter/CounterDemo.vue</code></strong>
      <ol>
        <li>找到 <code>count</code>：它的初始值是什么？为什么用 <code>ref</code>？</li>
        <li>读 <code>increment → decrement → reset</code>：每个函数修改了什么？</li>
        <li>切到模板，从 <code>@click</code> 找回函数，再找出读取 <code>count</code> 的位置。</li>
      </ol>
      <p>读完试着解释：点击按钮 → 修改 count.value → 大数字和按钮状态一起更新。</p>
    </div>
    <CodeExample :source="counterSource" filename="CounterDemo.vue" />
    <div class="explanation-grid">
      <article class="explanation-card">
        <span class="explanation-number">01</span><h3>声明一个会被追踪的值</h3>
        <code>const count = ref(0)</code>
        <p><code>ref(0)</code> 创建初始值为 0 的响应式对象。Vue 会追踪它的变化；普通变量 <code>let count = 0</code> 本身不具备这种能力。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">02</span><h3>把状态放到页面里</h3>
        <code v-pre>{{ count }}</code>
        <p>双大括号把数据渲染为文字。在模板里，顶层的 ref 会自动解包，直接写 <code>count</code>；在 JavaScript 中要写 <code>count.value</code>。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">03</span><h3>用事件改变状态</h3>
        <code>@click="increment"</code>
        <p><code>@click</code> 是 <code>v-on:click</code> 的简写。点击按钮会执行函数，函数修改 <code>count.value</code>，Vue 随后更新页面。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">04</span><h3>让属性也跟随数据</h3>
        <code>:disabled="count === 0"</code>
        <p><code>:</code> 是 <code>v-bind:</code> 的简写，让属性接收表达式的结果。注意：直接写 <code>disabled="false"</code> 仍会禁用按钮，因为这是 HTML 布尔属性。</p>
      </article>
    </div>
    <div class="learning-note"><strong>一个容易混淆的细节</strong><p><code>const</code> 固定的是 ref 对象的引用，不是它内部的值，所以 <code>count.value += 1</code> 完全合法。<code>&lt;script setup&gt;</code> 中声明的变量和函数可以直接在模板里使用。</p></div>
  </section>

  <section id="practice" class="lesson-section" aria-labelledby="practice-title">
    <div class="practice-panel">
      <div class="practice-header"><span class="pill">基础必做 · 3 个 TODO</span><span class="practice-symbol" aria-hidden="true">↗</span></div>
      <h2 id="practice-title">给计数器加上步长和边界</h2>
      <p>在编辑器打开 <code>src/lessons/01-counter/CounterPractice.vue</code>。每完成一个 TODO 就保存，在下方「我的练习」观察变化。</p>
      <ol class="practice-goals">
        <li><strong>TODO 1 · 按步长加减：</strong>使用已有的 <code>step</code>，同步按钮文字；减法不能小于 0。</li>
        <li><strong>TODO 2 · 限制上限：</strong>计数不能超过 <code>maxCount = 10</code>，到达上限时禁用加法按钮。</li>
        <li><strong>TODO 3 · 完整重置：</strong>计数变为 0，步长恢复为 2。</li>
      </ol>
      <p class="starter-note">起始代码每次仍加减 1，步长选择只改变 step 的值；上限限制和步长重置尚未实现。这些正是你要完成的任务。</p>
    </div>
    <div class="demo-panel practice-preview">
      <div class="panel-heading"><span><span class="status-dot"></span>我的练习 · 修改 Practice 文件后在这里验证</span><span>PRACTICE</span></div>
      <CounterPractice />
    </div>
    <details class="exercise-answer practice-source">
      <summary>查看我的练习源码（含 TODO）</summary>
      <div class="answer-content"><CodeExample :source="practiceSource" filename="CounterPractice.vue" /></div>
    </details>
    <p class="stretch-goal"><strong>进阶挑战：</strong>增加「加到上限」按钮，并用 <code>:disabled</code> 在达到上限时禁用它。先独立尝试，不影响基础任务验收。</p>
  </section>

  <section id="checklist" class="lesson-section" aria-labelledby="checklist-title">
    <div class="section-heading"><div><span class="section-kicker">VERIFY</span><h2 id="checklist-title">验证你写的功能</h2></div></div>
    <ManualChecklist :checks="checks" />
  </section>

  <section id="hints" class="lesson-section" aria-labelledby="hints-title">
    <div class="section-heading"><div><span class="section-kicker">WHEN YOU NEED HELP</span><h2 id="hints-title">卡住时，一次只看一点</h2></div></div>
    <details class="exercise-answer">
      <summary>提示 1 · 先理清思路</summary>
      <div class="answer-content"><p>把每次写死的 1 换成步长。先计算新值，再把它限制在 0 到 10 之间。按钮禁用是交互提示，函数内部也需要保护边界。</p></div>
    </details>
    <details class="exercise-answer">
      <summary>提示 2 · 关键语法</summary>
      <div class="answer-content">
        <p><code>Math.min(上限, 新值)</code> 防止超过上限，<code>Math.max(0, 新值)</code> 防止小于 0。</p>
        <p>脚本中读 <code>step.value</code>，模板中写 <code v-pre>+ {{ step }}</code>。重置需要修改两个 ref 的 <code>.value</code>。</p>
      </div>
    </details>
    <details class="exercise-answer full-solution">
      <summary>完整参考答案 · 完成后再对照</summary>
      <div class="answer-content">
        <p>这是基础任务的一种实现，文件位于 <code>01-counter/solutions/CounterSolution.vue</code>。下方答案演示独立运行，不会替换你的练习。</p>
        <CounterSolution />
        <CodeExample :source="solutionSource" filename="CounterSolution.vue" />
      </div>
    </details>
  </section>
  <div class="next-lesson">
    <span>完成后继续学习</span>
    <button class="text-button" type="button" @click="$emit('navigate', '02')">02 个人资料编辑器 →</button>
    <span>认识 reactive、v-model 和 computed</span>
  </div>
</template>
