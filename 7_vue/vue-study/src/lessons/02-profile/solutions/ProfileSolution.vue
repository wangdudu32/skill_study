<script setup>
import { computed, reactive } from 'vue'

const profile = reactive({
  name: 'Vue 学习者',
  role: '前端入门',
  bio: '每天写一点 Vue。',
})

const displayName = computed(() => profile.name.trim() || '未命名同学')
const bioLength = computed(() => profile.bio.length)

const nameError = computed(() => {
  const name = profile.name.trim()
  if (!name) return '请输入昵称。'
  if (name.length > 12) return '昵称最多 12 个字符。'
  return ''
})
const bioError = computed(() => bioLength.value > 80 ? '自我介绍最多 80 个字符。' : '')
// computed 可以依赖另一个 computed，在 JavaScript 中通过 .value 读取。
const isValid = computed(() => !nameError.value && !bioError.value)

function reset() {
  profile.name = 'Vue 学习者'
  profile.role = '前端入门'
  profile.bio = '每天写一点 Vue。'
}
</script>

<template>
  <div class="profile-editor">
    <form class="profile-form" @submit.prevent>
      <label for="solution-name">昵称</label>
      <input id="solution-name" v-model="profile.name" type="text" autocomplete="off" :aria-invalid="Boolean(nameError)" aria-describedby="solution-name-help" />
      <p id="solution-name-help" class="field-help" :class="{ 'field-error': nameError }">{{ nameError || '昵称去除首尾空白后需为 1—12 个字符。' }}</p>

      <label for="solution-role">学习阶段</label>
      <select id="solution-role" v-model="profile.role">
        <option>前端入门</option>
        <option>正在练习</option>
        <option>能写小项目</option>
      </select>

      <label for="solution-bio">自我介绍</label>
      <textarea id="solution-bio" v-model="profile.bio" rows="3" :aria-invalid="Boolean(bioError)" aria-describedby="solution-bio-help"></textarea>
      <p id="solution-bio-help" class="field-help" :class="{ 'field-error': bioError }">{{ bioError || '自我介绍最多 80 个字符，可以留空。' }} 已输入 {{ bioLength }} / 80</p>
      <button class="secondary-button" type="button" @click="reset">重置资料</button>
    </form>

    <aside class="profile-preview" aria-label="答案资料预览">
      <span class="small-label">完成后的资料预览</span>
      <h3>{{ displayName }}</h3>
      <span class="profile-role">{{ profile.role }}</span>
      <p class="profile-bio">{{ profile.bio || '还没有写自我介绍。' }}</p>
      <p class="validation-status" :class="{ invalid: !isValid }" role="status">{{ isValid ? '资料可用' : '请先修正输入' }}</p>
      <p class="preview-note">仅在当前页面预览，不会提交到服务器。</p>
    </aside>
  </div>
</template>
