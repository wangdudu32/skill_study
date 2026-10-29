<script setup>
import { ref } from 'vue'

// 先读 CounterDemo.vue，再按 TODO 顺序修改这个文件。
const count = ref(0)
const step = ref(2)
const maxCount = 10

function increment() {
  // TODO 1：使用 step.value 作为步长，并修改模板中的按钮文字。
  // TODO 2：把结果限制在 maxCount 以内；想想 9 + 2 应该是多少。
  count.value += 1
}

function decrement() {
  // TODO 1：按步长减少，并确保 1 - 2 的结果是 0，而不是 -1。
  if (count.value > 0) {
    count.value -= 1
  }
}

function reset() {
  count.value = 0
  // TODO 3：重置时也把 step 恢复为 2。
}
</script>

<template>
  <div class="counter-demo">
    <p class="counter-label">我的计数器</p>
    <output class="counter-value" aria-live="polite" aria-label="练习计数">{{ count }}</output>
    <div class="step-picker" role="group" aria-label="选择步长">
      <span>步长</span>
      <!-- 模板会自动解包 ref，这里可以直接给 step 赋值。 -->
      <button type="button" :aria-pressed="step === 1" @click="step = 1">1</button>
      <button type="button" :aria-pressed="step === 2" @click="step = 2">2</button>
      <button type="button" :aria-pressed="step === 3" @click="step = 3">3</button>
    </div>
    <div class="counter-actions">
      <!-- TODO 1：让两个按钮的文字显示当前 step。 -->
      <button class="counter-button" type="button" :disabled="count === 0" @click="decrement">− 1</button>
      <!-- TODO 2：达到 maxCount 时禁用加法按钮。函数内部也要限制上限。 -->
      <button class="counter-button primary" type="button" @click="increment">+ 1</button>
    </div>
    <button class="reset-button" type="button" @click="reset">重置计数和步长</button>
    <div class="state-inspector" aria-label="练习实时状态">
      <div><span>count.value</span><code>{{ count }}</code></div>
      <div><span>step.value</span><code>{{ step }}</code></div>
      <div><span>目标上限 maxCount</span><code>{{ maxCount }}</code></div>
    </div>
  </div>
</template>
