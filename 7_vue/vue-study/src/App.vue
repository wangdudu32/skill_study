<script setup>
import { computed, nextTick, ref, watchEffect } from 'vue'
import CounterLesson from './lessons/01-counter/CounterLesson.vue'
import ProfileLesson from './lessons/02-profile/ProfileLesson.vue'

// 这里是学习页面的外壳。初学时先阅读各课的 Demo.vue 和 Practice.vue。
const lessons = [
  { number: '01', title: '计数器', topic: '认识 Vue 的响应式', component: CounterLesson },
  { number: '02', title: '个人资料编辑器', topic: '双向绑定与计算属性', component: ProfileLesson },
]
const upcomingLessons = [
  { number: '03', title: '待办事项列表', topic: '条件渲染与列表渲染' },
  { number: '04', title: '商品卡片与购物车', topic: '组件之间如何通信' },
  { number: '05', title: '搜索用户', topic: '侦听器与异步操作' },
  { number: '06', title: '可挂载的计时器', topic: '生命周期与逻辑复用' },
]
const selectedLesson = ref('01')
const currentLesson = computed(() => lessons.find((lesson) => lesson.number === selectedLesson.value))
const mainContent = ref(null)

watchEffect(() => {
  document.title = `Vue 学习实验室 · ${currentLesson.value.title}`
})

async function selectLesson(number) {
  if (number === selectedLesson.value || !lessons.some((lesson) => lesson.number === number)) return
  selectedLesson.value = number
  // 新课程从头开始，避免沿用上一课的段落锚点。
  if (window.location.hash) {
    window.history.replaceState(null, '', window.location.pathname + window.location.search)
  }
  await nextTick()
  mainContent.value.focus({ preventScroll: true })
  mainContent.value.scrollIntoView({ behavior: 'instant', block: 'start' })
}
</script>

<template>
  <a class="skip-link" href="#main-content">跳到学习内容</a>
  <div class="app-layout">
    <aside class="sidebar">
      <a class="brand" href="#main-content" aria-label="Vue 学习实验室首页">
        <img src="./assets/logo.svg" width="32" height="32" alt="" />
        <span>Vue 学习实验室<small>一步一步，把概念变成实践</small></span>
      </a>
      <div class="sidebar-heading"><span>学习路线</span><span>6 个小实验</span></div>
      <nav class="lesson-nav" aria-label="学习案例">
        <button
          v-for="lesson in lessons"
          :key="lesson.number"
          type="button"
          class="lesson-link"
          :class="{ active: selectedLesson === lesson.number }"
          :aria-current="selectedLesson === lesson.number ? 'page' : undefined"
          @click="selectLesson(lesson.number)"
        >
          <span class="lesson-number">{{ lesson.number }}</span>
          <span class="lesson-name">{{ lesson.title }}<small>{{ lesson.topic }}</small></span>
          <span v-if="selectedLesson === lesson.number" class="active-dot" aria-hidden="true"></span>
        </button>
        <button
          v-for="lesson in upcomingLessons"
          :key="lesson.number"
          class="lesson-link upcoming"
          type="button"
          disabled
          :aria-label="`${lesson.title}，后续学习，尚未实现`"
        >
          <span class="lesson-number">{{ lesson.number }}</span>
          <span class="lesson-name">{{ lesson.title }}<small>{{ lesson.topic }}</small></span>
        </button>
      </nav>
      <p class="upcoming-note">已开放 2 / 6 课 · 03—06 后续开放。</p>
      <div class="sidebar-note">
        <span class="note-symbol" aria-hidden="true">{ }</span>
        <strong>先读源码，再亲手实现。</strong>
        <p>Demo 看示例，Practice 写功能。一次完成一个 TODO，再验证效果。</p>
        <a href="#practice">开始写练习 <span aria-hidden="true">↗</span></a>
      </div>
      <div class="sidebar-footer"><span class="status-dot"></span>Vue 3 · JavaScript</div>
    </aside>
    <div class="workspace">
      <header class="topbar">
        <span>学习路线 <span class="breadcrumb-divider">/</span> <strong>{{ currentLesson.number }} {{ currentLesson.title }}</strong></span>
        <span class="topbar-label">在实践中理解 Vue</span>
      </header>
      <main id="main-content" ref="mainContent" class="main-content" tabindex="-1">
        <p class="session-note">切换课程或刷新会重置演示和勾选记录；你保存到文件里的代码会保留。</p>
        <component :is="currentLesson.component" @navigate="selectLesson" />
        <footer class="page-footer">
          <span>每一个小实验，都是理解 Vue 的一步。</span>
          <a href="#main-content">回到顶部 ↑</a>
        </footer>
      </main>
    </div>
  </div>
</template>
