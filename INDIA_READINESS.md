# Telangana general-hospital readiness

Target: small general hospitals in Warangal and Karimnagar. This is a documentation adaptation, not hospital licensing, NABH accreditation, ABDM certification, or clinical validation.

## Implemented
The discharge form includes an opt-in India documentation checklist, facility and location details, treating clinician registration/council, optional patient contact, history/assessment summary, investigations, discharge condition, instruction language and patient-facing text, and applicable consent record references. When enabled and saved, missing required checklist items block finalization through the storage layer. Contacts for the patient remain optional. Existing encounters are not retrospectively marked compliant. Supplementary admission/discharge timestamps display IST while storage retains timezone-aware UTC. Fahrenheit entry remains supported. No ABHA or Aadhaar requirement is introduced.

The consent reference is documentation of an external record, not a substitute for informed consent or a patient signature. Instruction-language entry does not translate text. The hospital must verify that instructions are understandable to the patient. The checklist includes product design additions as well as fields informed by the sources below; it is not a verbatim statutory checklist.

## Before a real-patient pilot
- Confirm the facility's bed count, services and applicable Telangana registration requirements with its administrator and local legal adviser; do not assume all national standards apply identically.
- Replace the public shared demo with hospital-isolated storage, authenticated users, role permissions, session controls, access/event auditing, backup/restore testing and a documented retention policy.
- Establish recording notices, consent workflow where applicable, patient rights handling and contracts/data flows for each external processor, reviewed against applicable DPDP commencement dates.
- Validate speech on local accents and Telugu/English requirements. Current recognizer is English-only.
- Have registered clinicians/pharmacists verify Indian medication products and references. US labels cannot establish Indian product suitability.
- Assess CDSCO applicability for the intended risk/decision-support claims and validate models on appropriate data; synthetic performance is not clinical evidence.
- If required, implement and validate ABDM integration separately. No ABHA verification or FHIR interoperability certification is currently provided.
- Add hospital-approved pathways for transfer, LAMA/DAMA and death summaries before claiming complete discharge coverage. Current finalized discharge flow is not a death certification workflow.

## Sources reviewed
- CEA Allopathic Hospital Level 2, sections 10.19–10.20, 10.36–10.38 and Annexures 7–9: https://clinicalestablishments.mohfw.gov.in/sites/default/files/2022-06/885.pdf
- CEA applicability/state information: https://www.clinicalestablishments.mohfw.gov.in/
- MoHFW EHR Standards 2016: https://www.mohfw.gov.in/sites/default/files/EMR-EHR_Standards_for_India_as_notified_by_MOHFW_2016_0.pdf
- ABDM integration FAQs: https://abdm.gov.in/faqs

This file records remaining work explicitly; the current public deployment must continue to use fictional data.
