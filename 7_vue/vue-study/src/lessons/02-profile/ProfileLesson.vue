<script setup>
import CodeExample from '../../components/CodeExample.vue'
import ManualChecklist from '../../components/ManualChecklist.vue'
import ProfileDemo from './ProfileDemo.vue'
import ProfilePractice from './ProfilePractice.vue'
import ProfileSolution from './solutions/ProfileSolution.vue'
import demoSource from './ProfileDemo.vue?raw'
import practiceSource from './ProfilePractice.vue?raw'
import solutionSource from './solutions/ProfileSolution.vue?raw'

defineEmits(['navigate'])

const checks = [
  { id: 'binding', action: '昵称改为「小林」，阶段改为「正在练习」', expected: '预览立即显示小林和正在练习，不需要点击保存。' },
  { id: 'computed', action: '昵称输入「  小林  」，介绍改为「学习 Vue」', expected: '预览昵称为「小林」，字符数为 6；输入框仍保留昵称首尾空格。' },
  { id: 'empty', action: '昵称清空，再输入几个空格', expected: '两种情况都显示昵称错误，预览为「未命名同学」，状态为「请先修正输入」。' },
  { id: 'name-limit', action: '昵称输入 123456789012，再追加一个 3', expected: '12 个字符没有昵称错误，13 个字符出现昵称错误；删除最后一个字符后恢复。' },
  { id: 'bio-limit', action: '保持有效昵称；介绍输入 80 个 a，再追加一个 a', expected: '80 个字符时资料可用，81 个字符时提示介绍超长；删回 80 后恢复可用。' },
  { id: 'reset', action: '点击重置资料，再清空自我介绍', expected: '重置时恢复默认资料和字数；介绍可以留空，字数为 0，资料仍然可用。' },
]
</script>

<template>
  <section class="lesson-intro" aria-labelledby="lesson-title">
    <div class="eyebrow"><span class="pill">实验 02</span><span>表单入门 · 建议 35 分钟</span></div>
    <h1 id="lesson-title">编辑一份资料，<br class="mobile-break" />理解<span>双向绑定。</span></h1>
    <p class="intro-description">输入框改变对象，对象驱动预览。再用计算属性，让昵称、字数和校验结果自动跟上变化。</p>
    <div class="concept-tags" aria-label="本节知识点">
      <span><code>reactive</code> 响应式对象</span>
      <span><code>v-model</code> 双向绑定</span>
      <span><code>computed</code> 计算属性</span>
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
    <div class="section-heading"><div><span class="section-kicker">PLAYGROUND</span><h2 id="demo-title">输入一次，观察两处变化</h2></div><span class="section-caption">先看绑定和预览，校验留给练习</span></div>
    <div class="demo-panel">
      <div class="panel-heading"><span><span class="status-dot"></span>参考演示 · 基础资料编辑器</span><span>DEMO</span></div>
      <ProfileDemo />
    </div>
    <div class="reading-guide observation-guide">
      <strong>带着三个问题操作</strong>
      <ol>
        <li>修改昵称和学习阶段，右边的预览为什么也会改变？</li>
        <li>把昵称删空，为什么预览变成了「未命名同学」？</li>
        <li>输入自我介绍，字数是在哪里计算的？有没有手动监听输入事件？</li>
      </ol>
    </div>
  </section>

  <section id="source" class="lesson-section" aria-labelledby="source-title">
    <div class="section-heading"><div><span class="section-kicker">READ THE SOURCE</span><h2 id="source-title">从对象，一路读到输入框</h2></div></div>
    <div class="reading-guide">
      <strong>先读这个文件：<code>src/lessons/02-profile/ProfileDemo.vue</code></strong>
      <ol>
        <li>找到 <code>reactive({ ... })</code>：把三个字段与页面上的输入框对应起来。</li>
        <li>找到模板中的三个 <code>v-model</code>：沿着绑定找到它们修改的属性。</li>
        <li>回到 <code>computed</code>：找出 <code>displayName</code> 和 <code>bioLength</code> 各自依赖哪个字段。</li>
      </ol>
      <p>数据流：输入 → profile 的属性改变 → computed 重新计算 → 预览更新。</p>
    </div>
    <CodeExample :source="demoSource" filename="ProfileDemo.vue" />
    <div class="explanation-grid">
      <article class="explanation-card">
        <span class="explanation-number">01</span><h3>把相关状态放在一起</h3>
        <code>const profile = reactive({ ... })</code>
        <p><code>reactive</code> 让对象的属性可被追踪。脚本和模板都写 <code>profile.name</code>，不需要 <code>profile.value.name</code>。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">02</span><h3>让输入和状态互相同步</h3>
        <code>v-model="profile.name"</code>
        <p><code>:value</code> 只负责把状态显示到输入框。<code>v-model</code> 还会把用户输入写回状态；它适用于这里的输入框、下拉框和文本域。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">03</span><h3>派生值交给计算属性</h3>
        <code>computed(() => profile.bio.length)</code>
        <p>字数依赖自我介绍，没必要再用一个独立变量手动同步。<code>computed</code> 会跟踪依赖并缓存结果，依赖变化后再更新。</p>
      </article>
      <article class="explanation-card">
        <span class="explanation-number">04</span><h3>区分原始输入与显示结果</h3>
        <code>profile.name.trim() || '未命名同学'</code>
        <p>这里得到的是处理后的显示值，不会修改输入框中的原文。计算属性中只计算和返回值，不要顺手修改它依赖的状态。</p>
      </article>
    </div>
    <div class="learning-note"><strong>和第一课连起来</strong><p><code>computed</code> 返回的也是 ref：模板直接使用，脚本中用 <code>.value</code>。暂时不要把 <code>profile.name</code> 解构成普通变量，也不要替换整个对象；先通过属性读写状态。</p></div>
  </section>

  <section id="practice" class="lesson-section" aria-labelledby="practice-title">
    <div class="practice-panel">
      <div class="practice-header"><span class="pill">基础必做 · 3 个 TODO</span><span class="practice-symbol" aria-hidden="true">↗</span></div>
      <h2 id="practice-title">让这份资料真正动起来</h2>
      <p>打开 <code>src/lessons/02-profile/ProfilePractice.vue</code>。保留已有界面，先连通输入，再写派生值，最后加校验。</p>
      <ol class="practice-goals">
        <li><strong>TODO 1 · 双向绑定：</strong>把三个字段的 <code>:value</code> 改成 <code>v-model</code>，观察学习阶段和自我介绍预览。</li>
        <li><strong>TODO 2 · 实时计算：</strong>用 <code>computed</code> 实现去空白后的昵称、空昵称兜底和介绍字数。</li>
        <li><strong>TODO 3 · 输入校验：</strong>昵称去除首尾空白后为 1—12 个字符，介绍不超过 80 个字符且可留空；实现 <code>nameError</code>、<code>bioError</code> 和 <code>isValid</code>。</li>
      </ol>
      <p class="starter-note">起始代码只做单向显示：输入还不会写回 profile，昵称、字数和校验是占位值。下方「资料可用」在完成 TODO 3 前不代表校验正确。</p>
    </div>
    <div class="demo-panel practice-preview">
      <div class="panel-heading"><span><span class="status-dot"></span>我的练习 · 保存后在这里观察变化</span><span>PRACTICE</span></div>
      <ProfilePractice />
    </div>
    <details class="exercise-answer practice-source">
      <summary>查看我的练习源码（含 TODO）</summary>
      <div class="answer-content"><CodeExample :source="practiceSource" filename="ProfilePractice.vue" /></div>
    </details>
    <p class="stretch-goal"><strong>进阶挑战：</strong>新增一个「城市」字段，用双向绑定编辑，再通过计算属性生成「昵称 · 城市」的个人签名。试试城市为空时的显示方式。</p>
    <p class="lesson-footnote">本课用 JavaScript 的 <code>String.length</code> 计数，先用普通汉字和英文字母验证；部分 emoji 会占多个计数单位。边界测试可在控制台运行 <code>'a'.repeat(80)</code> 生成测试文本。</p>
  </section>

  <section id="checklist" class="lesson-section" aria-labelledby="checklist-title">
    <div class="section-heading"><div><span class="section-kicker">VERIFY</span><h2 id="checklist-title">从正常输入测到边界情况</h2></div></div>
    <ManualChecklist :checks="checks" />
  </section>

  <section id="hints" class="lesson-section" aria-labelledby="hints-title">
    <div class="section-heading"><div><span class="section-kicker">WHEN YOU NEED HELP</span><h2 id="hints-title">先看思路，再对照实现</h2></div></div>
    <details class="exercise-answer">
      <summary>提示 1 · 先把数据连起来</summary>
      <div class="answer-content"><p>只改 TODO 1 时，学习阶段和介绍就应该实时更新。昵称还显示占位文字是正常的，它需要 TODO 2 的计算属性。每完成一步就检查对应结果。</p></div>
    </details>
    <details class="exercise-answer">
      <summary>提示 2 · 校验也是派生状态</summary>
      <div class="answer-content">
        <p>用 <code>computed(() => ...)</code> 包住计算过程。昵称校验先取 <code>profile.name.trim()</code>，再判断空值和长度，合法时返回 <code>''</code>。</p>
        <p>介绍长度已经有 <code>bioLength</code>，在脚本中通过 <code>bioLength.value</code> 复用。是否可用可以写成 <code>!nameError.value &amp;&amp; !bioError.value</code>。</p>
      </div>
    </details>
    <details class="exercise-answer full-solution">
      <summary>完整参考答案 · 完成后再对照</summary>
      <div class="answer-content">
        <p>这是基础任务的一种实现，位于 <code>02-profile/solutions/ProfileSolution.vue</code>。可在下方试验空值和长度边界，再与自己的代码对照。</p>
        <ProfileSolution />
        <CodeExample :source="solutionSource" filename="ProfileSolution.vue" />
      </div>
    </details>
  </section>
  <div class="next-lesson">
    <span>下一课：03 待办事项列表 · 后续开放</span>
    <strong>接下来学习列表渲染、条件渲染和数组更新</strong>
    <button class="text-button" type="button" @click="$emit('navigate', '01')">← 回到第一课复习</button>
  </div>
</template>
