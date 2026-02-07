```
Adityas-Laptop:ai-agentic-rag averma$ uv run ragchain evaluate
Evaluating 20 questions...

[1/20] What is Python used for?

[2/20] Compare Go and Rust for systems programming

[3/20] What are the key features of functional programming in Haskell?

[4/20] How has Java evolved since its release?

[5/20] What are the main differences between interpreted and compiled languages?

[6/20] Which languages are commonly used for machine learning?

[7/20] What are the top 10 most popular languages?

[8/20] How does TypeScript differ from JavaScript?

[9/20] What is the primary purpose of C# and the .NET framework?

[10/20] Why is C still preferred over C++ for embedded systems?

[11/20] What are the main use cases for PHP in modern web development?

[12/20] Why is SQL classified as a domain-specific language?

[13/20] Compare Swift and Objective-C for iOS development

[14/20] What role does Ruby on Rails play in web development?

[15/20] How does R differ from Python for statistical analysis?

[16/20] Why did Google adopt Kotlin as the preferred language for Android?

[17/20] Why is Fortran still used in scientific computing?

[18/20] What are the primary industries that still use COBOL?

[19/20] What makes Scratch distinct from text-based programming languages?

[20/20] Why is Ada used in safety-critical systems like aerospace?
2026-02-07 07:26:53,117 - chromadb.telemetry.product.posthog - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2026-02-07 07:26:58,412 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:26:58,426 - ragchain.inference.grader - INFO - Grade: YES (doc 3 rank 1, score=1.000, MRR=1.000)
2026-02-07 07:26:58,427 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: What is Python used for?...
2026-02-07 07:27:01,060 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:27:30,869 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:27:36,563 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:27:39,113 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:27:39,120 - ragchain.inference.grader - INFO - Grade: YES (doc 1 rank 1, score=0.650, MRR=1.000)
2026-02-07 07:27:39,121 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: Compare Go and Rust for systems programming...
2026-02-07 07:27:50,092 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:28:42,989 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:28:50,449 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:28:54,699 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:28:54,745 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:28:54,746 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: What are the key features of functional programmin...
2026-02-07 07:29:08,098 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:29:43,880 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:29:51,454 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:29:53,437 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:29:53,944 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.825, MRR=1.000)
2026-02-07 07:29:53,945 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: How has Java evolved since its release?...
2026-02-07 07:30:08,022 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:30:49,702 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:30:55,470 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:30:59,886 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:30:59,936 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.767, MRR=1.000)
2026-02-07 07:30:59,938 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: What are the main differences between interpreted ...
2026-02-07 07:31:12,442 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:31:45,120 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:31:51,328 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:31:55,384 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:31:55,459 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:31:55,459 - ragchain.evaluation.judge - INFO - Retrieved 6 docs for: Which languages are commonly used for machine lear...
2026-02-07 07:32:09,366 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:32:40,594 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:32:50,143 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:32:53,177 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:32:53,217 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=1.000, MRR=1.000)
2026-02-07 07:32:53,218 - ragchain.evaluation.judge - INFO - Retrieved 6 docs for: What are the top 10 most popular languages?...
2026-02-07 07:33:07,323 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:33:33,335 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:33:39,610 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:33:39,713 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.767, MRR=1.000)
2026-02-07 07:33:39,713 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: How does TypeScript differ from JavaScript?...
2026-02-07 07:33:48,958 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:34:27,651 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:34:33,714 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:34:35,534 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:34:35,552 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.650, MRR=1.000)
2026-02-07 07:34:35,552 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: What is the primary purpose of C# and the .NET fra...
2026-02-07 07:34:48,881 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:35:23,878 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:35:32,369 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:35:37,112 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:35:37,366 - ragchain.inference.grader - INFO - Grade: YES (doc 1 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:35:37,369 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: Why is C still preferred over C++ for embedded sys...
2026-02-07 07:35:49,819 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:36:41,434 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:36:49,679 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:36:51,375 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:36:51,388 - ragchain.inference.grader - INFO - Grade: YES (doc 1 rank 1, score=0.700, MRR=1.000)
2026-02-07 07:36:51,389 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: What are the main use cases for PHP in modern web ...
2026-02-07 07:37:02,315 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:37:40,462 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:37:48,061 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:37:48,099 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:37:48,100 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: Why is SQL classified as a domain-specific languag...
2026-02-07 07:37:57,481 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:38:29,427 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:38:35,487 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:38:37,285 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:38:37,301 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:38:37,302 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: Compare Swift and Objective-C for iOS development...
2026-02-07 07:38:49,154 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:39:38,179 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:39:44,239 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:39:46,085 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:39:46,098 - ragchain.inference.grader - INFO - Grade: YES (doc 1 rank 1, score=0.650, MRR=1.000)
2026-02-07 07:39:46,099 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: What role does Ruby on Rails play in web developme...
2026-02-07 07:39:58,768 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:40:19,170 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:40:26,213 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:40:28,624 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:40:28,739 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.650, MRR=1.000)
2026-02-07 07:40:28,740 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: How does R differ from Python for statistical anal...
2026-02-07 07:40:39,564 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:41:02,684 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:41:10,787 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:41:12,193 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:41:12,224 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.883, MRR=1.000)
2026-02-07 07:41:12,225 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: Why did Google adopt Kotlin as the preferred langu...
2026-02-07 07:41:24,721 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:41:53,980 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:42:00,425 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:42:00,442 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.860, MRR=1.000)
2026-02-07 07:42:00,443 - ragchain.evaluation.judge - INFO - Retrieved 5 docs for: Why is Fortran still used in scientific computing?...
2026-02-07 07:42:14,023 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:42:46,392 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:42:52,411 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:42:53,924 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:42:53,950 - ragchain.inference.grader - INFO - Grade: YES (doc 3 rank 1, score=0.720, MRR=1.000)
2026-02-07 07:42:53,951 - ragchain.evaluation.judge - INFO - Retrieved 6 docs for: What are the primary industries that still use COB...
2026-02-07 07:43:09,899 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:43:41,655 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:43:49,334 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:43:51,181 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:43:51,194 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.700, MRR=1.000)
2026-02-07 07:43:51,195 - ragchain.evaluation.judge - INFO - Retrieved 4 docs for: What makes Scratch distinct from text-based progra...
2026-02-07 07:44:01,705 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:44:37,032 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:44:43,719 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:44:46,346 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
2026-02-07 07:44:46,422 - ragchain.inference.grader - INFO - Grade: YES (doc 0 rank 1, score=0.700, MRR=1.000)
2026-02-07 07:44:46,423 - ragchain.evaluation.judge - INFO - Retrieved 3 docs for: Why is Ada used in safety-critical systems like ae...
2026-02-07 07:44:55,803 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2026-02-07 07:45:32,900 - httpx - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"

==================================================
EVALUATION SUMMARY
==================================================

Q1: What is Python used for?...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Python is used for a variety of purposes, as explicitly stated in the context:  
1. **Machine learni...

Q2: Compare Go and Rust for systems programming...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 4/5
  Answer: **Comparison of Go and Rust for Systems Programming**  

| **Feature**               | **Go**       ...

Q3: What are the key features of functional programmin...
  Correctness: 4/5, Relevance: 5/5, Faithfulness: 3/5
  Answer: The key features of functional programming in Haskell, as stated in the context, include:  

1. **Pu...

Q4: How has Java evolved since its release?...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Java has evolved significantly since its release in May 1995. Initially designed for interactive tel...

Q5: What are the main differences between interpreted ...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: The main differences between interpreted and compiled languages, as stated in the context, are as fo...

Q6: Which languages are commonly used for machine lear...
  Correctness: 2/5, Relevance: 1/5, Faithfulness: 1/5
  Answer: Based on the provided context, **Python** is explicitly stated as a language commonly used for machi...

Q7: What are the top 10 most popular languages?...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Based on the provided context, I cannot find specific information about the top 10 most popular prog...

Q8: How does TypeScript differ from JavaScript?...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: TypeScript and JavaScript differ in several key aspects, as outlined in the context:  

1. **Static ...

Q9: What is the primary purpose of C# and the .NET fra...
  Correctness: 2/5, Relevance: 2/5, Faithfulness: 1/5
  Answer: Based on the provided context, I cannot find specific information about the primary purpose of C# an...

Q10: Why is C still preferred over C++ for embedded sys...
  Correctness: 2/5, Relevance: 1/5, Faithfulness: 1/5
  Answer: ...

Q11: What are the main use cases for PHP in modern web ...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Based on the provided context, the main use cases for PHP in modern web development are:  

1. **Web...

Q12: Why is SQL classified as a domain-specific languag...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: SQL is classified as a domain-specific language (DSL) because it is **designed specifically for mana...

Q13: Compare Swift and Objective-C for iOS development...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: **Comparison of Swift and Objective-C for iOS Development**  

| **Feature**                     | *...

Q14: What role does Ruby on Rails play in web developme...
  Correctness: 2/5, Relevance: 1/5, Faithfulness: 5/5
  Answer: Based on the provided context, I cannot find specific information about Ruby on Rails or its role in...

Q15: How does R differ from Python for statistical anal...
  Correctness: 2/5, Relevance: 2/5, Faithfulness: 3/5
  Answer: Based on the provided context, I cannot find specific information about how R differs from Python fo...

Q16: Why did Google adopt Kotlin as the preferred langu...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 3/5
  Answer: Based on the provided context, the specific reasons for Google's adoption of Kotlin as the preferred...

Q17: Why is Fortran still used in scientific computing?...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Fortran remains widely used in scientific computing due to several key factors explicitly stated in ...

Q18: What are the primary industries that still use COB...
  Correctness: 2/5, Relevance: 1/5, Faithfulness: 1/5
  Answer: The primary industries that still use COBOL, as stated in the context, are:  
- **Business**  
- **F...

Q19: What makes Scratch distinct from text-based progra...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Scratch is distinct from text-based programming languages due to its **block-based, visual interface...

Q20: Why is Ada used in safety-critical systems like ae...
  Correctness: 5/5, Relevance: 5/5, Faithfulness: 5/5
  Answer: Ada is used in safety-critical systems like aerospace **because of its robust error-detection mechan...

Average Scores:
  Correctness: 4.05/5
  Relevance: 3.90/5
  Faithfulness: 3.85/5
Adityas-Laptop:ai-agentic-rag averma$ 
```
