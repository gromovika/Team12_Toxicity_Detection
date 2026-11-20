# Как безопасно перенести подготовленные файлы в командный GitHub через VS Code (macOS)

**Важно:** эта версия собрана на основе присланного ZIP и проверенных локальных результатов. Прямой доступ к текущим удалённым коммитам и правам участников GitHub отсутствовал. Сначала сравните её с актуальной веткой команды; **не заменяйте основной репозиторий вслепую**.

## Вариант через отдельную ветку и Pull Request (предпочтительно)

1. Сохраните локальные `.env`, `data/dataset.txt.zip`, результаты MLflow и любые непубликуемые файлы **вне GitHub**. Не копируйте их в новую папку для публикации.
2. В терминале VS Code клонируйте командный репозиторий, если он ещё не клонирован как git-репозиторий:

   ```bash
   git clone https://github.com/gromovika/Team12_Toxicity_Detection.git
   cd Team12_Toxicity_Detection
   git checkout -b submission/readme-mlops
   ```

3. Распакуйте `Team12_Repo_Ready_for_GitHub.zip` в **отдельную** папку. В VS Code сравните файлы и перенесите нужные изменения (главный `README.md`, `docs/`, `.env.example`, `compose.yaml`, `scripts/local_validate.sh`, обновлённые ноутбуки и тесты). При наличии новых коммитов у других участников **объедините**, а не заменяйте их работу. Для более простого переноса воспользуйтесь `Team12_Repo_Docs_and_Fixes_Patch.zip` (содержит только изменённые файлы).
4. Проверьте изменения:

   ```bash
   git status --short
   git diff --stat
   git check-ignore -v .env data/dataset.txt.zip
   git add .
   git diff --cached --name-only
   ```

   Проверьте, что **нет** `.env`, `data/`, персональных текстов, токенов и файлов, права на публикацию которых не подтверждены.
5. Повторите тесты в клоне и проверьте `README.md`, Mermaid и внутренние ссылки. После code review:

   ```bash
   git commit -m "docs: prepare MLOps repository for assessment"
   git push -u origin submission/readme-mlops
   ```

   Откройте Pull Request, получите одобрение команды и только затем объедините с веткой сдачи. Если нет прав записи в репозиторий, попросите владельца принять patch или пригласить вас как collaborator.
6. Откройте GitHub Actions и дождитесь зелёного CI. Убедитесь, что преподаватель имеет доступ к ветке/репозиторию, а ссылки из README работают. **Репозиторий не был обновлён автоматически в этом чате.**

## Если хотите продолжать локальную проверку на Mac

У вас уже была рабочая папка `Team12_Toxicity_Detection_Local_Checks`, где порт `8000` оказался занят. В новой версии задайте в существующем **локальном** `.env` строку:

```dotenv
API_HOST_PORT=18000
```

Новая `compose.yaml` автоматически пробросит `127.0.0.1:18000 → api:8000`, а `scripts/local_validate.sh` использует `18000` в health и status checks. Не переписывайте `.env` из архива и не удаляйте Docker volumes. Для проверки новой конфигурации:

```bash
bash -n scripts/local_validate.sh
docker compose --profile mlflow --profile test config --quiet
bash scripts/local_validate.sh infra
bash scripts/local_validate.sh tests
```

После переноса в git смотрите только тот pipeline, который реально запущен на вашей машине; скриншоты более ранних проверок не заменяют нового запуска изменённой версии.
