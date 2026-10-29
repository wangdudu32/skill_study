<script setup>
import { computed, reactive } from 'vue'

// 多个相关字段放在同一个响应式对象中；读写属性不需要 .value。
const profile = reactive({
  name: 'Vue 学习者',
  role: '前端入门',
  bio: '每天写一点 Vue。',
})

// 从已有状态推导显示内容，不再维护一份可能忘记同步的数据。
// computed 在依赖变化后重新计算；回调里只返回结果，不修改 profile。
const displayName = computed(() => profile.name.trim() || '未命名同学')
const bioLength = computed(() => profile.bio.length)

function reset() {
  // 保留 reactive 对象，修改它的属性。
  profile.name = 'Vue 学习者'
  profile.role = '前端入门'
  profile.bio = '每天写一点 Vue。'
}
</script>

<template>
  <div class="profile-editor">
    <form class="profile-form" @submit.prevent>
      <label for="demo-name">昵称</label>
      <!-- v-model 同时负责把状态显示到输入框，以及把输入写回状态。 -->
      <input id="demo-name" v-model="profile.name" type="text" autocomplete="off" />

      <label for="demo-role">学习阶段</label>
      <select id="demo-role" v-model="profile.role">
        <option>前端入门</option>
        <option>正在练习</option>
        <option>能写小项目</option>
      </select>

      <label for="demo-bio">自我介绍</label>
      <textarea id="demo-bio" v-model="profile.bio" rows="3"></textarea>
      <p class="field-help">已输入 {{ bioLength }} 个字符</p>
      <button class="secondary-button" type="button" @click="reset">重置资料</button>
    </form>

    <aside class="profile-preview" aria-label="参考资料预览">
      <span class="small-label">实时资料预览</span>
      <!-- computed 也是 ref，模板中会自动解包。 -->
      <h3>{{ displayName }}</h3>
      <span class="profile-role">{{ profile.role }}</span>
      <p class="profile-bio">{{ profile.bio || '还没有写自我介绍。' }}</p>
      <p class="preview-note">仅在当前页面预览，不会提交到服务器。</p>
    </aside>
  </div>
</template>
