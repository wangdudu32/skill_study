<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  source: { type: String, required: true },
  filename: { type: String, required: true },
})
const selectedSection = ref('script')
const sections = [
  { id: 'script', label: '逻辑 · script' },
  { id: 'template', label: '模板 · template' },
  { id: 'full', label: '完整组件' },
]

// 展示实际组件文件的内容，修改案例后，页面上的代码也会同步更新。
const displayedSource = computed(() => {
  if (selectedSection.value === 'script') {
    return props.source.match(/<script\b[^>]*>([\s\S]*?)<\/script>/)?.[1].trim() ?? props.source
  }
  if (selectedSection.value === 'template') {
    // 顶层结束标签独占一行，避免把嵌套的 <template> 当作组件的结束。
    return props.source.match(/^<template\b[^>]*>([\s\S]*?)^<\/template>/m)?.[1].trim() ?? props.source
  }
  return props.source.trim()
})
</script>

<template>
  <div class="code-example">
    <div class="code-toolbar">
      <div class="code-switcher" role="group" aria-label="选择源码片段">
        <button
          v-for="section in sections"
          :key="section.id"
          type="button"
          :class="{ selected: selectedSection === section.id }"
          :aria-pressed="selectedSection === section.id"
          @click="selectedSection = section.id"
        >{{ section.label }}</button>
      </div>
      <span class="code-filename">{{ filename }}</span>
    </div>
    <pre class="source-code" tabindex="0" :aria-label="`${filename} 源码`"><code>{{ displayedSource }}</code></pre>
    <div class="code-footer"><span class="status-dot"></span>实际文件内容 · 保存源码后同步更新</div>
  </div>
</template>
