# Contributing Guidelines

Thank you for your interest in contributing to MMROS 🚀  
We appreciate your effort to help improve this project. Please follow the guidelines below to ensure a smooth collaboration.

## How to Contribute

### 1. Reporting Issues

If you encounter any bugs, have feature requests, or want to ask questions, please [open an issue](https://github.com/ktro2828/mmros/issues).

When reporting an issue, please include:

- A clear and descriptive title.
- A detailed explanation of the problem.
- Steps to reproduce the issue (if applicable).
- Relevant error messages or logs.
- Your environment (OS, ROS version, etc.)

### 2. Submitting Pull Requests

We welcome pull requests (PRs) to address issues, improve documentation, or add new features.  
Here's the process:

1. **Fork the repository** and clone your fork:

   ```bash
   git clone https://github.com/<YOUR-USERNAME>/awviz-ros.git
   cd awviz-ros
   ```

   To ensure changes align with the existing code style, use [pre-commit](https://pre-commit.com/).  
   For the installation, please refer to the official document.

   Before to start making your changes, please run the following command to set up `pre-commit` hooks:

   ```bash
   pre-commit install
   ```

   Now, `pre-commit` will run automatically on `git commit`!

2. **Create a new branch** for your contribution:

   ```bash
   git checkout -b feat/<PACKAGE-NAME>/<YOUR-FEATURE-NAME>
   ```

3. **Make your changes**, ensuring they align with the existing code style. Remember to update or add relevant tests.

4. **Commit your changes** with clear adn descriptive commit messages:

   Note that, we basically follow the [Conventional Commits](https://www.conventionalcommits.org/).

   ```bash
   git add <PATH-TO-CHANGES>
   git commit -sS -m "feat: <YOUR FEATURE>"
   ```

5. **Push your branch** to your fork:

   ```bash
   git push <YOUR-REMOTE> feat/<PACKAGE-NAME>/<YOUR-FEATURE-NAME>
   ```

6. **Submit a pull request** to the main repository. Please describe your changes in detail, linking to any relevant issues.

### 3. Coding Rules

To maintain a clean and consistent codebase, please follow these guidelines:

- **Code Style**
  - Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for general style conventions.
  - Use `snake_case` for function names. For example, `void set_name(const std::string & name);`.
- **Code Formatting**
  - We recommend using [clangd](https://clangd.llvm.org/) that features like code completion, navigation (e.g. go to definition, find references), refactoring, and diagnostics.
- **Code Documentation**
  - Ensure your code is well-documented code with clear and concise comments.
  - Follow the [Doxygen C++ Document Style](https://www.doxygen.nl/manual/index.html) for writing code documentation.

### 4. Testing

Before submitting a pull request, ensure that all tests pass successfully. You can run the tests using:

```bash
colcon test --packages-select awviz awviz_common awviz_plugin [<PACKAGES>..]
```

If you introduce a new feature or fix a bug, please add appropriate tests to cover the changes.

### 5. Documentation

Well-written documentation is crucial for both users and developers. If your contribution affects the behavior of the system, please ensure that:

- All public functions are documented.
- New features are described in the README or relevant documentation files.

### 6. License

By contributing to `awviz-ros`, you agree that your contributions will be licensed under the project's @ref LICENSE "Apache-2.0".

## Add New Project

When adding a new project, please create a new ROS 2 package under the `projects` directory:

```shell
cd projects
ros2 create pkg <PROJECT_NAME>
```

Please add the description of the new project under the `docs/projects/<PROJECT_NAME>.md` following the [TEMPLATE](./projects/TEMPLATE.md).
After finishing create the documentation, link it to `README.md` of the new package.

```shell
ln -s docs/projects/<PROJECT_NAME>.md projects/<PROJECT_NAME>/README.md
```

## Get in Touch

If you have any questions, feel free to reach out by opening an issue. We're happy to assist!
