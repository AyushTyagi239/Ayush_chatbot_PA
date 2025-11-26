# Create directory structure
$folders = @(
    "knowledge-base/about",
    "knowledge-base/education",
    "knowledge-base/experience",
    "knowledge-base/skills",
    "knowledge-base/projects/ai_llm",
    "knowledge-base/projects/game_dev",
    "knowledge-base/projects/web_dev",
    "knowledge-base/extras"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

# Create files with basic structure
$files = @(
    "knowledge-base/about/personal_details.md",
    "knowledge-base/about/bio.md",
    "knowledge-base/about/tagline.md",
    "knowledge-base/about/contact_links.md",

    "knowledge-base/education/school.md",
    "knowledge-base/education/college.md",
    "knowledge-base/education/certifications.md",

    "knowledge-base/experience/intensity_global.md",
    "knowledge-base/experience/tara_application_internship.md",

    "knowledge-base/skills/languages.md",
    "knowledge-base/skills/frontend.md",
    "knowledge-base/skills/backend.md",
    "knowledge-base/skills/ai_llm.md",
    "knowledge-base/skills/tools.md",
    "knowledge-base/skills/game_engines.md",

    "knowledge-base/projects/ai_llm/code_converter.md",
    "knowledge-base/projects/ai_llm/multimodal_assistant.md",
    "knowledge-base/projects/ai_llm/smart_deal_notifier.md",

    "knowledge-base/projects/game_dev/mountain_rider.md",
    "knowledge-base/projects/game_dev/soldierio.md",
    "knowledge-base/projects/game_dev/space_shooter_ranger.md",
    "knowledge-base/projects/game_dev/car_simulator.md",

    "knowledge-base/projects/web_dev/foodies_hub.md",
    "knowledge-base/projects/web_dev/food_website_react.md",
    "knowledge-base/projects/web_dev/timerace.md",
    "knowledge-base/projects/web_dev/investment_calculator.md",
    "knowledge-base/projects/web_dev/react_tictactoe.md",

    "knowledge-base/extras/interests.md",
    "knowledge-base/extras/achievements.md",
    "knowledge-base/extras/future_goals.md"
)

foreach ($file in $files) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

# Count total created markdown files
$count = (Get-ChildItem -Path "knowledge-base" -Recurse -Filter "*.md").Count

Write-Host "Knowledge base directory structure created!"
Write-Host "Total .md files created: $count"
