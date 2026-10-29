<script setup>
import { ref } from 'vue'

defineProps({
  checks: { type: Array, required: true },
})

// 只记录学习者的手动确认，不会根据演示操作自动判定代码是否正确。
const completedChecks = ref([])
</script>

<template>
  <div class="manual-checklist">
    <p class="checklist-status">
      <span>操作「我的练习」，符合预期后再勾选。</span>
      <span class="progress-badge" aria-live="polite">{{ completedChecks.length }} / {{ checks.length }} 手动确认</span>
    </p>
    <div class="checklist">
      <label
        v-for="(check, index) in checks"
        :key="check.id"
        class="check-row"
        :class="{ checked: completedChecks.includes(check.id) }"
      >
        <input v-model="completedChecks" type="checkbox" :value="check.id" />
        <span class="check-copy">
          <strong><span class="check-number">{{ index + 1 }}.</span>{{ check.action }}</strong>
          <span>{{ check.expected }}</span>
        </span>
      </label>
    </div>
    <p v-if="completedChecks.length === checks.length" class="completion-message" role="status">
      你已手动确认全部项目。试着不看答案，解释一次「状态 → 事件 → 页面变化」。这些勾选不代表自动测试通过。
    </p>
  </div>
</template>
